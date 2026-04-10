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

    def fit(self, training_df):
        """Fit encoders and scalers on training data only.

        Call this on the training split before calling ``transform()`` on
        train, test, or scoring data.  This prevents test-set statistics
        from leaking into the fitted transformers.

        Args:
            training_df: DataFrame used to learn encoder categories and
                scaler min/max values.

        Returns:
            self (for chaining).
        """
        for column in training_df.columns:
            if column in self.var_types["numeric"]:
                scaler = MinMaxScaler()
                non_na = training_df[column].notna()
                if non_na.sum() > 0:
                    scaler.fit(training_df.loc[non_na, [column]])
                    self.min_max_scalers[column] = scaler
                # If all values are NaN in training, skip — column passes through unscaled
            elif column in self.var_types["categorical"]:
                encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                encoder.fit(training_df[[column]])
                self.one_hot_encoders[column] = encoder
        self._is_fitted = True
        return self

    def transform(self, df):
        """Apply previously-fitted encoders/scalers to *df*.

        Must call ``fit()`` first (or use ``vectorize_table()`` for the
        legacy fit-and-transform-in-one-step behaviour).

        Args:
            df: DataFrame to transform (train, test, or scoring data).

        Returns:
            Transformed DataFrame.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                "Table2Vector.transform() called before fit(). "
                "Call fit(training_df) first, or use vectorize_table() "
                "for the legacy fit_transform behaviour."
            )
        return self._apply_transforms(df)

    def vectorize_table(self, original_df, base_df=None):
        """Fit *and* transform in a single call (legacy API).

        When *base_df* is provided the encoders are fitted on *base_df*
        and applied to *original_df*.  Otherwise both fit and transform
        use *original_df*.

        .. note::

           Prefer the explicit ``fit()`` / ``transform()`` pair in
           training pipelines to avoid data leakage.

        Returns:
            Transformed DataFrame.
        """
        fit_df = base_df if base_df is not None else original_df
        self.fit(fit_df)
        return self._apply_transforms(original_df)

    # ------------------------------------------------------------------
    # Internal helper shared by transform() and vectorize_table()
    # ------------------------------------------------------------------
    def _apply_transforms(self, original_df):
        """Apply fitted encoders/scalers to *original_df*.

        Index alignment: one-hot encoded columns are assigned the same
        index as the source DataFrame so that ``pd.concat`` never
        introduces NaN values from mismatched indices.
        """
        vectorized_df = original_df.copy()

        for column in vectorized_df.columns:
            if column in self.var_types["numeric"] and column in self.min_max_scalers:
                scaler = self.min_max_scalers[column]
                non_na_rows = vectorized_df[column].notna()
                if non_na_rows.sum() > 0:
                    vectorized_df.loc[non_na_rows, column] = scaler.transform(
                        vectorized_df.loc[non_na_rows, [column]]
                    ).ravel()
            elif column in self.var_types["categorical"]:
                encoder = self.one_hot_encoders[column]
                df_encoded = pd.DataFrame(
                    encoder.transform(vectorized_df[[column]]),
                    index=vectorized_df.index,  # preserve index to avoid NaN on concat
                )
                df_encoded.columns = [
                    f"{column}{self.SEP}{cat}" for cat in encoder.categories_[0]
                ]
                vectorized_df = pd.concat([vectorized_df, df_encoded], axis=1)
                vectorized_df = vectorized_df.drop(column, axis=1)

        return vectorized_df

    def get_cardinalities(self, columns):
        """Return per-column cardinalities consistent with the fitted encoders.

        For categorical columns the cardinality equals the number of
        categories the fitted ``OneHotEncoder`` knows about.  For numeric
        columns the cardinality is 1 (a single scaled value).

        This must be called **after** ``fit()`` or ``vectorize_table()``
        so that the encoders are available.

        Args:
            columns: Ordered column names from the *cleaned* (pre-vectorized)
                DataFrame — the same order used when calling ``fit()``.

        Returns:
            List[int] of per-column cardinalities.
        """
        cardinalities = []
        for col in columns:
            if col in self.one_hot_encoders:
                cardinalities.append(len(self.one_hot_encoders[col].categories_[0]))
            elif col in self.min_max_scalers:
                cardinalities.append(1)
            else:
                # Column was not transformed — treat as single numeric value
                cardinalities.append(1)
        return cardinalities

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
                one_hot_encoder.inverse_transform(onehot),
                columns=[column],
                index=df.index,  # preserve index to avoid NaN on concat
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
