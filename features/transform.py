import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class Table2Vector:
    """
    Class for transforming data for machine learning.

    This class handles transformations like one-hot encoding for categorical data,
    min-max scaling for numerical data, and handling missing data.

    This class does not handle textual data or datetime variabls.
    """

    # Ignore for now. We will use it later
    VAR_TYPES = [
        "categorical",
        "numeric",
        "datetime",
        "text",
        "binary",
        "missing_indicator",  # indicator variable for missing values in another column
    ]

    def __init__(self, variable_types):
        """Initialize the transformer with the variable types dictionary."""
        self.SEP = "__"
        self.MISSING = "MISSING__"

        self.var_types = {
            "categorical": [],
            "numeric": [],
            "datetime": [],
            "text": [],
            "binary": [],
            "missing_indicator": [],
        }

        for k in self.var_types:
            self.var_types[k] = [
                var for var, var_type in variable_types.items() if var_type == k
            ]

        self.one_hot_encoders = {}
        self.min_max_scalers = {}

    def vectorize_table(self, original_df, base_df=None):
        """
        Transform the dataframe according to the variable types.

        Categorical variables are one-hot encoded, numeric variables are min-max scaled,
        and missing values are replaced with dummy variables.

        Returns:
        - The transformed dataframe.
        - Dictionaries with fitted OneHotEncoders and MinMaxScalers for each column.
        """
        vectorized_df = original_df.copy()

        for column in vectorized_df.columns:
            # We use a MixMaxScaler for numeric variables
            if column in self.var_types["numeric"]:
                min_max_scaler = MinMaxScaler()
                non_na_rows = vectorized_df[column].notna()
                vectorized_df.loc[non_na_rows, column] = min_max_scaler.fit_transform(
                    vectorized_df.loc[non_na_rows, [column]]
                ).ravel()
                self.min_max_scalers[column] = min_max_scaler
            elif column in self.var_types["categorical"]:
                one_hot_encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                if base_df is None:
                    df_encoded = pd.DataFrame(
                        one_hot_encoder.fit_transform(vectorized_df[[column]])
                    )
                else:
                    one_hot_encoder.fit(base_df[[column]])
                    df_encoded = pd.DataFrame(
                        one_hot_encoder.transform(vectorized_df[[column]])
                    )
                df_encoded.columns = [
                    f"{column}{self.SEP}{cat}" for cat in one_hot_encoder.categories_[0]
                ]
                vectorized_df = pd.concat([vectorized_df, df_encoded], axis=1)
                vectorized_df = vectorized_df.drop(column, axis=1)
                self.one_hot_encoders[column] = one_hot_encoder

        return vectorized_df

    def add_missing_indicators(self, df):
        """
        Adds binary columns to the dataframe indicating the presence of missing values.

        For each column in the dataframe, this function adds a corresponding column
        with a binary indicator of whether the value in that row is missing (NaN).
        These new columns are named 'missing_<column_name>' and are appended to the dataframe.

        Args:
            df (pd.DataFrame): The input pandas DataFrame.

        Returns:
            result (pd.DataFrame): The DataFrame with added missing value indicator columns.
        """

        # Create DataFrame with indicator of missing values

        # We will create missing value indicators if
        # (a) there is no such missing value indicator already for the column and
        # (b) the column is not already a missing value indicator
        cols = [
            c
            for c in df.columns
            if not c.startswith(self.MISSING) and f"{self.MISSING}{c}" not in df.columns
        ]
        df_missing = pd.concat([df[c].isnull().astype(int) for c in cols], axis=1)
        df_missing.columns = [f"{self.MISSING}{c}" for c in cols]

        return df_missing

    @staticmethod
    def proba_to_onehot(proba):
        """Convert a vector of probabilities into a max-likelihood one-hot vector."""
        onehot = np.zeros_like(proba)
        onehot[np.arange(len(proba)), np.argmax(proba, axis=1)] = 1
        return onehot

    def tabularize_vector(self, vectorized_df, restore_missing_values=False):
        """
        Reverse the transformations applied to the dataframe.

        One-hot encoded categorical variables are decoded and min-max scaled numeric variables
        are inverse scaled.

        Returns the original dataframe.
        """
        df = vectorized_df.copy()

        for column in self.var_types["categorical"]:
            one_hot_encoder = self.one_hot_encoders[column]
            original_cols = [
                col for col in df.columns if col.startswith(f"{column}{self.SEP}")
            ]
            onehot_encoded = df[original_cols].values

            # Convert probabilities to one-hot encoding and perform inverse transformation
            onehot = self.proba_to_onehot(onehot_encoded)
            df_original = pd.DataFrame(
                one_hot_encoder.inverse_transform(onehot), columns=[column]
            )

            # Set original value to NaN for rows that become "missing"
            # df_original.replace('missing', np.nan, inplace=True)

            df = pd.concat([df.drop(original_cols, axis=1), df_original], axis=1)

        for column in self.var_types["numeric"]:
            min_max_scaler = self.min_max_scalers[column]
            non_na_rows = df[column].notna()
            inverse_transformed = min_max_scaler.inverse_transform(
                df.loc[non_na_rows, [column]]
            )
            df.loc[non_na_rows, column] = inverse_transformed.flatten()

        df = df.drop(
            [col for col in df.columns if col.startswith(self.MISSING)], axis=1
        )

        return df
