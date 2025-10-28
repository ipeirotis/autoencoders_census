import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Class to handle data loading and preprocessing for the project.
    """

    def __init__(self, drop_columns, rename_columns, columns_of_interest, additional_drop_columns=None, additional_rename_columns=None, additional_columns_of_interest=None):
        self.DROP_COLUMNS = drop_columns
        self.RENAME_COLUMNS = rename_columns
        self.COLUMNS_OF_INTEREST = columns_of_interest
        self.ADDITIONAL_DROP_COLUMNS = additional_drop_columns
        self.ADDITIONAL_RENAME_COLUMNS = additional_rename_columns
        self.ADDITIONAL_COLUMNS_OF_INTEREST = additional_columns_of_interest

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

    def load_pennycook_2(self):
        url = "data/Pennycook et al._Study 2.csv"
        df = self.load_original_data(url)
        df = df.select_dtypes(exclude=["object", "string"])
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

        for col in df.select_dtypes(include="float").columns:
            if (
                df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        return self.prepare_original_dataset(df, replacements={})

    def load_inattentive(self):
        url = "data/inattentive_users.csv"
        df = self.load_original_data(url)
        # df = df.select_dtypes(exclude=["object", "string"])

        for col in df.select_dtypes(include="float").columns:
            if (
                df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        return self.prepare_original_dataset(df, replacements={})

    def load_attention_check(self):
        url = "data/attention_check.csv"
        df = self.load_original_data(url)

        for col in df.select_dtypes(include="float").columns:
            if (
                df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        return self.prepare_original_dataset(df, replacements={})

    def load_pennycook(self):
        url1 = "data/Pennycook et al._Study 1.csv"
        df1 = self.load_original_data(url1)
        df1 = df1.select_dtypes(exclude=["object", "string"])

        self.DROP_COLUMNS = self.ADDITIONAL_DROP_COLUMNS
        self.RENAME_COLUMNS = self.ADDITIONAL_RENAME_COLUMNS
        self.COLUMNS_OF_INTEREST = self.ADDITIONAL_COLUMNS_OF_INTEREST
        url2 = "data/Pennycook et al._Study 2.csv"
        df2 = self.load_original_data(url2)
        df2 = df2.select_dtypes(exclude=["object", "string"])

        df = pd.concat([df1, df2], axis=0)
        df = df.reset_index(drop=True)

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

    def load_bot_bot_mturk(self):
        url = "data/Bot_Bot_Bot__MTURK.csv"
        df = self.load_original_data(url)

        df = df[pd.to_numeric(df["Q6"], errors='coerce').notnull()]

        df["Q2_1"] = pd.to_numeric(df["Q2_1"], errors='coerce')
        df['Q2_2'] = pd.to_numeric(df['Q2_2'], errors='coerce')
        df['Q2_3'] = pd.to_numeric(df['Q2_3'], errors='coerce')
        df['Q4_1'] = pd.to_numeric(df['Q4_1'], errors='coerce')
        df['Q4_2'] = pd.to_numeric(df['Q4_2'], errors='coerce')
        df['Q4_3'] = pd.to_numeric(df['Q4_3'], errors='coerce')
        df['Q5_1'] = pd.to_numeric(df['Q5_1'], errors='coerce')
        df['Q5_2'] = pd.to_numeric(df['Q5_2'], errors='coerce')
        df['Q5_3'] = pd.to_numeric(df['Q5_3'], errors='coerce')


        df["Q2_time_diff"] = df["Q2_2"] - df["Q2_1"]
        df["Q2_submit_diff"] = df["Q2_3"] - df["Q2_2"]

        df["Q4_time_diff"] = df["Q4_2"] - df["Q4_1"]
        df["Q4_submit_diff"] = df["Q4_3"] - df["Q4_2"]

        df["Q5_time_diff"] = df["Q5_2"] - df["Q5_1"]
        df["Q5_submit_diff"] = df["Q5_3"] - df["Q5_2"]

        df.drop(columns=["Q2_1", "Q2_2", "Q2_3", "Q4_1", "Q4_2", "Q4_3", "Q5_1", "Q5_2", "Q5_3", "Q6"], inplace=True)

        df = df.reset_index(drop=True)

        return self.prepare_original_dataset(df, replacements={})

    def load_data(self, dataset: str):
        if dataset == "sadc_2015":
            return self.load_2015()

        if dataset == "sadc_2017":
            return self.load_2017()

        if dataset == "pennycook_1":
            return self.load_pennycook_1()

        if dataset == "pennycook_2":
            return self.load_pennycook_2()

        if dataset == "pennycook":
            return self.load_pennycook()

        if dataset == "bot_bot_mturk":
            return self.load_bot_bot_mturk()

        if dataset == "inattentive":
            return self.load_inattentive()

        if dataset == "attention_check":
            return self.load_attention_check()

        if dataset == "moral_data":
            return self.load_moral_data()

        if dataset == "mturk_ethics":
            return self.load_mturk_ethics()

        if dataset == "public_opinion":
            return self.load_public_opinion()

        if dataset == "racial_data":
            return self.load_racial_data()

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

        # original_df = original_df[original_df["Condition"] == 4].drop(columns=["Condition"]).reset_index(drop=True)

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
                (df_copy[column] < -0.7) & (df_copy[column] >= -1.4) & (df_copy[column] != used_value),
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

    def find_outlier_data_sadc_2017(self, data, outlier_column):
        """
        Find outlier data in the dataset.
        """
        df, _ = self.load_data(data)

        # now I want the samples of the above characteristics CONCURRENTLY to have a new outlier column equal to 1 and the rest 0
        df[outlier_column] = 0
        conditions = [
            df["carrot_eating"] == "4 or more times per day",
            df["green _salad_eating"] == "4 or more times per day",
            df["fruit_juice_drinking"] == "4 or more times per day",
            df["fruit_eating"] == "4 or more times per day",
            df["stheight_cat"] == "top-extreme",
            df["stweight_cat"] == "top-extreme",
            df["stheight_cat"] == "bottom-extreme",
            df["stweight_cat"] == "bottom-extreme",
        ]

        df.loc[sum(conditions) >= 3, outlier_column] = 1

        return df[outlier_column]

    def load_moral_data(self):
        url = "data/moral_data.csv"
        df = self.load_original_data(url)

        for col in df.select_dtypes(include="float").columns:
            if (
                    df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        df = df[df["complete"] == 1]
        df.drop(columns=["complete"], inplace=True)

        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        return self.prepare_original_dataset(df, replacements={})

    def load_mturk_ethics(self):
        url = "data/ethics.csv"
        df = self.load_original_data(url)

        for col in df.select_dtypes(include="float").columns:
            if (
                    df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        return self.prepare_original_dataset(df, replacements={})

    def load_public_opinion(self):
        url = "data/public_opinion.csv"
        df = self.load_original_data(url)

        for col in df.select_dtypes(include="float").columns:
            if (
                    df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        # df = df[df["Progress"] == 100]
        # df.drop(columns=["Progress"], inplace=True)

        df = df.reset_index(drop=True)

        return self.prepare_original_dataset(df, replacements={})

    def load_racial_data(self):
        url = "data/racial_data.csv"
        df = self.load_original_data(url)

        for col in df.select_dtypes(include="float").columns:
            if (
                    df[col].dropna() % 1 == 0
            ).all():  # Check if all non-NaN values are whole numbers
                df[col] = df[col].astype("Int64")

        df = df[df["Finished"] == "1. True"]
        df.drop(columns=["Finished"], inplace=True)

        df.dropna(inplace=True)
        df = df.reset_index(drop=True)

        return self.prepare_original_dataset(df, replacements={})



