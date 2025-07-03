import unittest

from dataset.loader import DataLoader


class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize DataLoader object once for all tests
        cls.data_loader = DataLoader()

        # Load the original datasets from the URLs
        cls.df2015 = cls.data_loader.load_original_data(DataLoader.DATASET_URL_2015)
        cls.df2017 = cls.data_loader.load_original_data(DataLoader.DATASET_URL_2017)

        # Load the prepared datasets
        cls.project_data_2015, cls.var_types_2015 = cls.data_loader.load_2015()
        cls.project_data_2017, cls.var_types_2017 = cls.data_loader.load_2017()

    def test_original_data_shape(self):
        self.assertEqual(self.df2015.shape[1], 305)
        self.assertEqual(self.df2017.shape[1], 305)
        self.assertEqual(self.df2015.shape[1], self.df2017.shape[1])

    def test_original_data_rows(self):
        self.assertEqual(self.df2017.shape[0], 14765)
        self.assertEqual(self.df2015.shape[0], 15624)

    def test_prepared_data_rows(self):
        self.assertEqual(self.df2015.shape[0], self.project_data_2015.shape[0])
        self.assertEqual(self.df2017.shape[0], self.project_data_2017.shape[0])

    def test_prepared_data_columns(self):
        self.assertEqual(self.project_data_2015.shape[1], 98)
        self.assertEqual(self.project_data_2017.shape[1], 98)

    def test_cardinality_consistency(self):
        vars2015 = self.project_data_2015.describe().T
        vars2017 = self.project_data_2017.describe().T
        self.assertEqual(
            vars2015.merge(vars2017, how="outer", left_index=True, right_index=True)
            .query("unique_x != unique_y")
            .index.shape[0],
            0,
        )

    def test_cardinality_ranges(self):
        cardinalities = self.project_data_2015.describe().T.unique
        self.assertEqual(cardinalities.min(), 2)
        self.assertEqual(cardinalities.max(), 9)
        self.assertEqual(cardinalities.sum(), 609)
