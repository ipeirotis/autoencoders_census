import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from dataset.loader import DataLoader


class TestDataLoaderAPI(unittest.TestCase):
    """Tests using synthetic data — always runs, no external files needed."""

    def test_constructor_requires_arguments(self):
        with self.assertRaises(TypeError):
            DataLoader()

    def test_constructor_accepts_required_args(self):
        loader = DataLoader(
            drop_columns=[],
            rename_columns={},
            columns_of_interest=[],
        )
        self.assertIsInstance(loader, DataLoader)

    def test_load_original_data_reads_csv(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
            result = loader.load_original_data(path)
            self.assertEqual(result.shape, (3, 2))
            self.assertListEqual(list(result.columns), ["a", "b"])
        finally:
            os.unlink(path)

    def test_load_original_data_drops_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            loader = DataLoader(drop_columns=["b"], rename_columns={}, columns_of_interest=[])
            result = loader.load_original_data(path)
            self.assertNotIn("b", result.columns)
            self.assertEqual(result.shape[1], 2)
        finally:
            os.unlink(path)

    def test_load_original_data_renames_columns(self):
        df = pd.DataFrame({"old_name": [1, 2, 3]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            loader = DataLoader(
                drop_columns=[],
                rename_columns={"old_name": "new_name"},
                columns_of_interest=[],
            )
            result = loader.load_original_data(path)
            self.assertIn("new_name", result.columns)
            self.assertNotIn("old_name", result.columns)
        finally:
            os.unlink(path)

    def test_load_original_data_selects_columns_of_interest(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            path = f.name
        try:
            loader = DataLoader(
                drop_columns=[],
                rename_columns={},
                columns_of_interest=[0, 2],  # select columns by index
            )
            result = loader.load_original_data(path)
            self.assertEqual(result.shape[1], 2)
            self.assertIn("a", result.columns)
            self.assertIn("c", result.columns)
        finally:
            os.unlink(path)

    def test_detect_continuous_vars(self):
        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        n = 25
        df = pd.DataFrame({
            "cat_col": ["a", "b", "c", "a", "b"] * 5,
            "continuous_col": np.random.randn(n),
            "many_unique": list(range(n)),
        })
        # many_unique has 25 unique values (> threshold 20), continuous_col is float64
        result = loader.detect_continuous_vars(df)
        self.assertIn("continuous_col", result)
        self.assertIn("many_unique", result)
        self.assertNotIn("cat_col", result)

    def test_convert_to_categorical_static(self):
        df = pd.DataFrame({
            "x": np.random.randn(100),
        })
        result = DataLoader.convert_to_categorical(df, numeric_vars=["x"])
        self.assertNotIn("x", result.columns)
        self.assertIn("x_cat", result.columns)
        valid_cats = {"top-extreme", "high", "bottom-extreme", "low", "normal", "zero", "missing", "unknown"}
        self.assertTrue(set(result["x_cat"].unique()).issubset(valid_cats))


# Skip SADC regression tests when data files are absent (e.g. in CI)
_SADC_2017_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "sadc_2017only_national_full.csv"
)
_YRBS_SYNTHETIC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "yrbs_synthetic_pipeline", "outputs", "final_dataset.csv"
)

_SADC_FILES_EXIST = os.path.isfile(_SADC_2017_PATH) and os.path.isfile(_YRBS_SYNTHETIC_PATH)


@unittest.skipIf(not _SADC_FILES_EXIST, "SADC/YRBS data files not present")
class TestDataLoaderSADCRegression(unittest.TestCase):
    """Regression tests that require real SADC + YRBS synthetic data files."""

    @classmethod
    def setUpClass(cls):
        from utils import define_necessary_elements

        elements = define_necessary_elements("sadc_2017", [], {}, [])
        drop_columns, rename_columns, interest_columns = elements[0], elements[1], elements[2]
        additional_drop, additional_rename, additional_interest = elements[3], elements[4], elements[5]

        cls.loader = DataLoader(
            drop_columns=drop_columns,
            rename_columns=rename_columns,
            columns_of_interest=interest_columns,
            additional_drop_columns=additional_drop,
            additional_rename_columns=additional_rename,
            additional_columns_of_interest=additional_interest,
        )

    def test_load_2017_succeeds(self):
        project_data, var_types = self.loader.load_2017()
        self.assertGreater(len(project_data), 0)
        self.assertIsInstance(var_types, dict)
        self.assertTrue(all(v in ("numeric", "categorical") for v in var_types.values()))
