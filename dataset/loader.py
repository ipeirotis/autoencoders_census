import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Class to handle data loading and preprocessing for the project.
    """

    def __init__(self, drop_columns, rename_columns, columns_of_interest):
        self.DROP_COLUMNS = drop_columns
        self.RENAME_COLUMNS = rename_columns
        self.COLUMNS_OF_INTEREST = columns_of_interest

    def load_2015(self):
        url = "data/sadc_2015only_national.csv"
        df2015 = self.load_original_data(url)
        return self.prepare_original_dataset(
            df2015,
            replacements={
                "obese": {1: "obese", 2: "not obese"},
                "overweight": {1: "overweight", 2: "not overweight"},
            },
        )

    def load_2017(self):
        url = "data/sadc_2017only_national_full.csv"
        df2017 = self.load_original_data(url)
        return self.prepare_original_dataset(
            df2017,
            replacements={
                "obese": {1: "obese", 2: "not obese"},
                "overweight": {1: "overweight", 2: "not overweight"},
            },
        )

    def load_pennycook_1(self):
        url = "data/Pennycook et al._Study 1.csv"
        df = self.load_original_data(url)
        df = df.select_dtypes(exclude=["object", "string"])
        # df = df.dropna(subset=["SciKnow", "Income"])
        df[
            [
                "sharing_political",
                "sharing_sports",
                "sharing_celebrity",
                "sharing_science",
                "sharing_business",
                "sharing_other",
                "facebook",
                "twitter",
                "snapchat",
                "instagram",
                "whatsapp",
                "other",
                "news_criticism",
                "news_side",
            ]
        ] = df[
            [
                "sharing_political",
                "sharing_sports",
                "sharing_celebrity",
                "sharing_science",
                "sharing_business",
                "sharing_other",
                "facebook",
                "twitter",
                "snapchat",
                "instagram",
                "whatsapp",
                "other",
                "news_criticism",
                "news_side",
            ]
        ].fillna(
            0
        )

        for i in range(1, 16):
            df[f"Fake1_time_diff_{i}"] = df[f"Fake1_RT_2_{i}"] - df[f"Fake1_RT_1_{i}"]
            df[f"Fake1_submit_diff_{i}"] = df[f"Fake1_RT_3_{i}"] - df[f"Fake1_RT_2_{i}"]
            df[f"Fake1_clicks_{i}"] = df[f"Fake1_RT_4_{i}"]

            df[f"Real1_time_diff_{i}"] = df[f"Real1_RT_2_{i}"] - df[f"Real1_RT_1_{i}"]
            df[f"Real1_submit_diff_{i}"] = df[f"Real1_RT_3_{i}"] - df[f"Real1_RT_2_{i}"]
            df[f"Real1_clicks_{i}"] = df[f"Real1_RT_4_{i}"]

            df.drop(columns=[f"Fake1_RT_{j}_{i}" for j in range(1, 5)] + [f"Real1_RT_{j}_{i}" for j in range(1, 5)], inplace=True)


        # convert to integer columns
        for col in df.select_dtypes(include="float").columns:
            if (
                df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        return self.prepare_original_dataset(df, replacements={})

    def load_eval_dataset(self, dataset):
        df = pd.read_csv(dataset)
        base_df, types = self.load_2017()
        return (df, base_df), types

    def load_data(self, dataset: str):
        if dataset == "sadc_2015":
            return self.load_2015()

        if dataset == "sadc_2017":
            return self.load_2017()

        if dataset == "pennycook_1":
            return self.load_pennycook_1()

        return self.load_eval_dataset(dataset)

    def load_original_data(self, dataset_url):
        try:
            original_df = pd.read_csv(dataset_url, encoding="utf-8")
        except UnicodeDecodeError:
            original_df = pd.read_csv(dataset_url, encoding="latin1")

        if self.DROP_COLUMNS:
            original_df = original_df.drop(columns=self.DROP_COLUMNS)

        if self.COLUMNS_OF_INTEREST:
            original_df = original_df.iloc[:, self.COLUMNS_OF_INTEREST]

        if self.RENAME_COLUMNS:
            original_df = original_df.rename(columns=self.RENAME_COLUMNS)

        return original_df

    @staticmethod
    def convert_to_categorical(df, numeric_vars):
        df_copy = df.copy()

        for column in numeric_vars:
            missing_mask = df_copy[column].isna()

            scaler = StandardScaler()
            df_copy[column] = scaler.fit_transform(df_copy[[column]])

            highest_value = list(df_copy[column].value_counts().items())[0]
            if highest_value[1] > 0.5 * len(df_copy):
                used_value = highest_value[0]

            else:
                used_value = 0

            conditions = [
                (df_copy[column] > 1.4) & (df_copy[column] != used_value),
                (df_copy[column] <= 1.4) & (df_copy[column] > 0.7) & (df_copy[column] != used_value),
                (df_copy[column] < -1.4) & (df_copy[column] != used_value),
                (df_copy[column] < -0.7) & (df_copy[column] > -1.4) & (df_copy[column] != used_value),
                (missing_mask),
                (df_copy[column] >= -0.7) & (df_copy[column] <= 0.7) & (df_copy[column] != used_value),
                (df_copy[column] == used_value),
            ]
            choices = [
                "top-extreme",
                "high",
                "bottom-extreme",
                "low",
                "missing",
                "normal",
                "zero",
            ]
            df_copy[column + "_cat"] = np.select(conditions, choices, default="unknown")

        df_copy = df_copy.drop(columns=numeric_vars)

        return df_copy

    def detect_continuous_vars(self, df):
        """
        Detect continuous variables in the dataset.
        """
        threshold = 20
        continuous_columns = [
            column
            for column in df.columns
            if df[column].nunique() > threshold or df[column].dtype == "float64"
        ]

        return continuous_columns

    def prepare_original_dataset(self, project_data, replacements):
        """
        Prepare the dataset for the project, this includes selecting specific columns, renaming them
        and categorizing them as numeric or categorical.
        """

        for k, v in replacements.items():
            project_data[k] = project_data[k].replace(v)

        numeric_vars = self.detect_continuous_vars(project_data)
        project_data = DataLoader.convert_to_categorical(project_data, numeric_vars)

        categorical_vars = [c for c in project_data.columns if c not in numeric_vars]
        for c in categorical_vars:
            project_data[c] = project_data[c].astype("str")

        variable_types = {
            column: ("numeric" if column in numeric_vars else "categorical")
            for column in project_data.columns
        }

        return project_data, variable_types

    def find_outlier_data(self, data, outlier_column):
        """
        Find outlier data in the dataset.
        """
        df, _ = self.load_data(data)

        return df[outlier_column]
