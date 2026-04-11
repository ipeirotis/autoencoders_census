import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import warnings
from pandas.errors import PerformanceWarning

# Silence the gramentation warning so the terminal stays clean
warnings.simplefilter(action='ignore', category=PerformanceWarning)

# Default upper bound on the number of unique values a column may have to
# survive the "Rule of 9" cardinality filter. Columns with more than this
# many unique values are treated as too high-cardinality for the
# autoencoder and dropped; columns with <= 1 unique values are also
# dropped because they carry no signal. This threshold is configurable
# per-DataLoader via the ``max_unique_values`` constructor argument and
# via the CLI / YAML config (TASKS.md 3.1); the constant is re-exported
# so downstream modules can reference the same default.
DEFAULT_MAX_UNIQUE_VALUES = 9


class DataLoader:
    """
    Class to handle data loading and preprocessing for the project.
    """

    def __init__(
        self,
        drop_columns,
        rename_columns,
        columns_of_interest,
        additional_drop_columns=None,
        additional_rename_columns=None,
        additional_columns_of_interest=None,
        max_unique_values=None,
        apply_rule_of_n=True,
    ):
        self.DROP_COLUMNS = drop_columns
        self.RENAME_COLUMNS = rename_columns
        self.COLUMNS_OF_INTEREST = columns_of_interest
        self.ADDITIONAL_DROP_COLUMNS = additional_drop_columns
        self.ADDITIONAL_RENAME_COLUMNS = additional_rename_columns
        self.ADDITIONAL_COLUMNS_OF_INTEREST = additional_columns_of_interest
        # Rule-of-N threshold: ``None`` means use the module default.
        # Stored as an attribute so every loader method that ultimately
        # calls ``prepare_original_dataset`` (e.g. ``load_2017``,
        # ``load_uploaded_csv``) inherits the caller's configured
        # threshold without needing to thread the value through each
        # dataset-specific entry point (TASKS.md 3.1).
        if max_unique_values is None:
            max_unique_values = DEFAULT_MAX_UNIQUE_VALUES
        if max_unique_values < 2:
            raise ValueError(
                f"max_unique_values must be >= 2 (got {max_unique_values}); "
                "a threshold below 2 would drop every column since the "
                "Rule-of-N filter also rejects columns with <= 1 unique "
                "values."
            )
        self.max_unique_values = int(max_unique_values)
        # When ``False``, ``prepare_original_dataset`` skips the
        # Rule-of-N filter entirely — numeric binning, NaN-fill, and
        # string casting still run, but every surviving column is kept
        # regardless of cardinality. Scoring paths set this to ``False``
        # when a saved training-time vectorizer is present so that the
        # vectorizer (not the loader) is the authoritative source of
        # truth on which columns the model expects. Without this,
        # columns whose training-time cardinality exceeded the current
        # loader threshold would be silently dropped here and then
        # backfilled as constant ``"missing"`` values by
        # ``_clean_for_saved_vectorizer``, corrupting model inputs
        # (Codex P1 PR#49).
        self.apply_rule_of_n = bool(apply_rule_of_n)

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
        """Load an arbitrary CSV for evaluation against the 2017 SADC baseline.

        Returns the same ``(DataFrame, metadata)`` format as all other loaders.
        The raw evaluation DataFrame is read from *dataset*; preprocessing
        (binning, Rule-of-9 filtering) is applied via ``prepare_original_dataset``.
        """
        df = self.load_original_data(dataset)
        return self.prepare_original_dataset(df, replacements={})

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

    def load_data(self, dataset: str, csv_bytes: bytes = None):
        # new "uploaded" branch
        if dataset == "uploaded":
            if csv_bytes is None:
                raise ValueError("csv_bytes must be provided for uploaded dataset")
            return self.load_uploaded_csv(csv_bytes)
        
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
    
    def load_uploaded_csv(self, csv_bytes: bytes, replacements=None):
        """
        Specific entry point for frontend uploads
        
        Args:
            csv_bytes: The raw file content as bytes
            replacements: Optional dictionary for column name replacements
        """
        if replacements is None:
            replacements = {}
            
        # load raw data
        df = self.load_original_data(csv_bytes)
        
        # run preprocessing
        return self.prepare_original_dataset(df, replacements=replacements)

    def load_original_data(self, dataset_source, encoding=None):
        """
        Loads data from a file path (str) or raw bytes (uploaded file).

        Args:
            dataset_source: str (file path), bytes (raw upload), or file-like.
            encoding: Optional explicit encoding to try first. Callers that
                already detected an encoding (e.g. worker.py validate_csv()
                via chardet) pass it here so parsing uses the same codec as
                validation and a cp1252-encoded CSV is not silently decoded
                as Latin-1 or UTF-8, which would corrupt non-ASCII
                categoricals like smart quotes and em dashes and shift the
                downstream outlier scores.
        """
        # 1. Handle Raw Bytes (frontend upload)
        if isinstance(dataset_source, bytes):
            # Build the list of encodings to try: caller-provided first,
            # then the historical UTF-8 -> Latin-1 fallback chain.
            candidates = []
            if encoding:
                candidates.append(encoding)
            for fallback in ("utf-8", "latin1"):
                if fallback not in candidates:
                    candidates.append(fallback)

            last_err = None
            original_df = None
            for candidate in candidates:
                try:
                    original_df = pd.read_csv(
                        io.BytesIO(dataset_source), encoding=candidate
                    )
                    break
                except UnicodeDecodeError as err:
                    last_err = err
                    continue
            if original_df is None:
                raise last_err  # exhausted fallbacks

        # 2. Handle File-like objects
        elif isinstance(dataset_source, io.IOBase):
            original_df = pd.read_csv(dataset_source)

        else:
            candidates = []
            if encoding:
                candidates.append(encoding)
            for fallback in ("utf-8", "latin1"):
                if fallback not in candidates:
                    candidates.append(fallback)

            last_err = None
            original_df = None
            for candidate in candidates:
                try:
                    original_df = pd.read_csv(dataset_source, encoding=candidate)
                    break
                except UnicodeDecodeError as err:
                    last_err = err
                    continue
            if original_df is None:
                raise last_err
                
        # colummn processing logic
        if self.DROP_COLUMNS:
            original_df = original_df.drop(columns=self.DROP_COLUMNS, errors='ignore')

        if self.COLUMNS_OF_INTEREST:
            # COLUMNS_OF_INTEREST may contain integer positional indices
            # or string column names — handle both.
            int_indices = [c for c in self.COLUMNS_OF_INTEREST if isinstance(c, int)]
            str_names = [c for c in self.COLUMNS_OF_INTEREST if isinstance(c, str)]

            selected = []
            if int_indices:
                valid = [i for i in int_indices if i < len(original_df.columns)]
                selected.extend(original_df.columns[valid].tolist())
            if str_names:
                selected.extend([c for c in str_names if c in original_df.columns])

            original_df = original_df[selected]

        if self.RENAME_COLUMNS:
            original_df = original_df.rename(columns=self.RENAME_COLUMNS)

        # original_df = original_df[original_df["Condition"] == 4].drop(columns=["Condition"]).reset_index(drop=True)

        return original_df

    @staticmethod
    def convert_to_categorical(df, numeric_vars):
        df_copy = df.copy()

        for column in numeric_vars:
            # 1. safety check
            # check if column is empty or has all the same value before scaling
            if df_copy[column].nunique(dropna=True) <= 1:
                df_copy[column + "_cat"] = "NA"
                
                # drop the original bad column
                df_copy = df_copy.drop(columns=[column])
                continue
                
            missing_mask = df_copy[column].isna()
            
            # 2. Scale
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df_copy[[column]])
            df_copy[column] = scaled_values
            
            # 3. Binning Logic
            counts = df_copy[column].value_counts()
            highest_value = list(counts.items())[0]
            # highest_value = list(df_copy[column].value_counts().items())[0]
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

        # 4. final cleanup
        # ignore errors because some columns in numeric_vars might have already been dropped inside the loop above
        df_copy = df_copy.drop(columns=numeric_vars, errors='ignore')

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

    def prepare_original_dataset(
        self,
        project_data,
        replacements,
        max_unique_values=None,
        apply_rule_of_n=None,
    ):
        """
        1. Bins numeric data (making it categorical)
        2. Fills remaining NaN values in categorical columns with "missing"
        3. Applies the Rule of N: keep columns with 2..N unique values,
           drop anything with <= 1 or > N unique values. ``N`` defaults to
           ``self.max_unique_values`` (historically 9) but can be
           overridden per-call via ``max_unique_values`` — CLI callers
           usually set it on the DataLoader instead.
        4. Returns cleaned dataframe and metadata (ignored_columns +
           variable_types)

        When ``apply_rule_of_n`` is ``False`` (either per-call or via
        the loader instance attribute), step 3 is skipped: every column
        is kept regardless of cardinality, still cast to string. Scoring
        paths that load a trained vectorizer use this so the vectorizer
        (not the loader) is the authoritative source of truth on which
        columns the model expects.

        This mirrors the inline cleaning in ``worker.py`` and the shared
        ``main.prepare_for_categorical`` helper so that the upload path
        produces the same clean frame as the CLI / worker paths.
        """
        if max_unique_values is None:
            max_unique_values = self.max_unique_values
        if max_unique_values < 2:
            raise ValueError(
                f"max_unique_values must be >= 2 (got {max_unique_values})"
            )
        if apply_rule_of_n is None:
            apply_rule_of_n = self.apply_rule_of_n

        # Apply replacements
        for k, v in replacements.items():
            if k in project_data.columns: # ensure code doesn't crash if col doesn't exist in current dataset
                project_data[k] = project_data[k].replace(v)

        # 1. Identify and Bin Numeric Columns
        # Only treat as numeric if the col is truly numeric
        numeric_vars = []
        for col in project_data.columns:
            if pd.api.types.is_numeric_dtype(project_data[col]):
                numeric_vars.append(col)

        project_data = DataLoader.convert_to_categorical(project_data, numeric_vars) # Convert numeric columns into categorical bins

        # 2. Fill NaN in non-numeric columns with the literal string
        # "missing" so that (a) the Rule-of-N count below treats missing as
        # a real category and (b) downstream ``astype(str)`` does not turn
        # NaN into the string "nan". ``convert_to_categorical`` already
        # handles NaN for numeric columns via its "missing" bin, so this
        # is a no-op on the newly-created ``*_cat`` columns.
        project_data = project_data.fillna("missing")

        kept_columns = []
        ignored_columns = []

        if not apply_rule_of_n:
            # Scoring path with a saved vectorizer: skip cardinality
            # filtering entirely and let the vectorizer decide which
            # columns the model expects. We still cast to string so
            # downstream one-hot encoding receives consistent dtypes.
            for col in project_data.columns:
                project_data[col] = project_data[col].astype(str)
                kept_columns.append(col)
        else:
            # 3. Rule of N: keep columns with 2..N unique values.
            # Both extremes are dropped — > N values is too high-cardinality
            # for the autoencoder to learn a meaningful one-hot, and a
            # single unique value provides no signal at all.
            for col in project_data.columns:
                n_unique = project_data[col].nunique(dropna=True)

                if 1 < n_unique <= max_unique_values:
                    kept_columns.append(col)
                    # kept as string for the Autoencoder
                    project_data[col] = project_data[col].astype(str)
                else:
                    if n_unique <= 1:
                        reason = "Low cardinality (<= 1 unique value)"
                    else:
                        reason = f"High cardinality (> {max_unique_values} unique values)"
                    ignored_columns.append({
                        "name": col,
                        "unique_values": int(n_unique),
                        "reason": reason,
                    })

        # 4. Construct Final dataframe
        clean_df = project_data[kept_columns].copy()

        # 5. Prepare Metadata
        variable_types = {c: "categorical" for c in clean_df.columns}

        metadata = {
            "ignored_columns": ignored_columns,
            "variable_types": variable_types
        }

        return clean_df, metadata

    def find_outlier_data(self, data, outlier_column):
        """
        Load dataset and extract gold-label columns for evaluation.

        Temporarily disables COLUMNS_OF_INTEREST filtering so that
        attention-check / screening columns are available even though
        they are intentionally excluded from the training config.
        """
        saved = self.COLUMNS_OF_INTEREST
        self.COLUMNS_OF_INTEREST = []
        try:
            df, _ = self.load_data(data)
        finally:
            self.COLUMNS_OF_INTEREST = saved

        return df[outlier_column]

    def find_outlier_data_sadc_2017(self, data, outlier_column):
        """
        Find outlier data in the SADC 2017 dataset.

        Temporarily disables COLUMNS_OF_INTEREST filtering so that
        all columns (including those used to build the composite
        outlier indicator) are available.
        """
        saved = self.COLUMNS_OF_INTEREST
        self.COLUMNS_OF_INTEREST = []
        try:
            df, _ = self.load_data(data)
        finally:
            self.COLUMNS_OF_INTEREST = saved

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



