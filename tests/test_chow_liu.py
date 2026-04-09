import unittest

import numpy as np
import pandas as pd

from chow_liu_rank import CLTree, ColumnSpec, rank_rows_by_chow_liu


def make_correlated_survey(n_rows=500, seed=42):
    """Generate synthetic survey data with strong inter-column correlations.

    Columns q1-q3 are correlated (q2 depends on q1, q3 depends on q2).
    Column q4 is independent noise.
    """
    rng = np.random.RandomState(seed)

    q1 = rng.choice(["a", "b", "c"], size=n_rows, p=[0.5, 0.3, 0.2])

    # q2 strongly depends on q1
    q2 = []
    for v in q1:
        if v == "a":
            q2.append(rng.choice(["x", "y"], p=[0.9, 0.1]))
        elif v == "b":
            q2.append(rng.choice(["x", "y"], p=[0.2, 0.8]))
        else:
            q2.append(rng.choice(["x", "y"], p=[0.5, 0.5]))

    # q3 depends on q2
    q3 = []
    for v in q2:
        if v == "x":
            q3.append(rng.choice(["low", "med", "high"], p=[0.7, 0.2, 0.1]))
        else:
            q3.append(rng.choice(["low", "med", "high"], p=[0.1, 0.3, 0.6]))

    # q4 is independent
    q4 = rng.choice(["yes", "no"], size=n_rows)

    return pd.DataFrame({"q1": q1, "q2": q2, "q3": q3, "q4": q4})


class TestCLTreeFit(unittest.TestCase):
    """Test CLTree.fit() on synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.df = make_correlated_survey(n_rows=500, seed=42)
        cls.tree = CLTree(alpha=1.0).fit(cls.df)

    def test_fit_sets_columns(self):
        self.assertEqual(self.tree.columns, ["q1", "q2", "q3", "q4"])

    def test_fit_builds_schema_for_all_columns(self):
        for col in self.df.columns:
            self.assertIn(col, self.tree.specs)
            spec = self.tree.specs[col]
            self.assertIsInstance(spec, ColumnSpec)
            self.assertGreater(spec.K, 0)

    def test_fit_sets_root(self):
        self.assertIsNotNone(self.tree.root)
        self.assertIn(self.tree.root, self.df.columns)

    def test_fit_builds_tree_with_correct_edges(self):
        edges = self.tree.edges()
        # A tree with 4 nodes has exactly 3 edges
        self.assertEqual(len(edges), 3)
        for u, v, mi in edges:
            self.assertIn(u, self.df.columns)
            self.assertIn(v, self.df.columns)
            self.assertGreaterEqual(mi, 0.0)

    def test_every_node_has_parent_except_root(self):
        for col in self.df.columns:
            if col == self.tree.root:
                self.assertIsNone(self.tree.parent[col])
            else:
                self.assertIsNotNone(self.tree.parent[col])
                self.assertIn(self.tree.parent[col], self.df.columns)

    def test_root_has_marginal_distribution(self):
        self.assertIn(self.tree.root, self.tree.p_root)
        p = self.tree.p_root[self.tree.root]
        self.assertAlmostEqual(p.sum(), 1.0, places=5)
        self.assertTrue((p > 0).all())

    def test_cpt_rows_sum_to_one(self):
        for (child, parent), table in self.tree.cpt.items():
            row_sums = table.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_fit_with_explicit_root(self):
        tree = CLTree(alpha=1.0).fit(self.df, root="q4")
        self.assertEqual(tree.root, "q4")

    def test_fit_with_mi_subsample(self):
        tree = CLTree(alpha=1.0).fit(self.df, mi_subsample=100)
        self.assertEqual(len(tree.edges()), 3)
        self.assertIsNotNone(tree.root)


class TestCLTreeLogLikelihood(unittest.TestCase):
    """Test CLTree.log_likelihood() produces valid per-row scores."""

    @classmethod
    def setUpClass(cls):
        cls.df = make_correlated_survey(n_rows=500, seed=42)
        cls.tree = CLTree(alpha=1.0).fit(cls.df)
        cls.logp = cls.tree.log_likelihood(cls.df)

    def test_returns_array_of_correct_length(self):
        self.assertEqual(len(self.logp), len(self.df))

    def test_all_log_likelihoods_are_negative(self):
        # Log-probabilities should always be <= 0
        self.assertTrue((self.logp <= 0).all())

    def test_no_nan_or_inf(self):
        self.assertFalse(np.any(np.isnan(self.logp)))
        self.assertFalse(np.any(np.isinf(self.logp)))

    def test_typical_rows_score_higher_than_random(self):
        """Rows from the fitted distribution should score higher than random rows."""
        rng = np.random.RandomState(99)
        random_df = pd.DataFrame({
            "q1": rng.choice(["a", "b", "c"], size=100),
            "q2": rng.choice(["x", "y"], size=100),
            "q3": rng.choice(["low", "med", "high"], size=100),
            "q4": rng.choice(["yes", "no"], size=100),
        })
        random_logp = self.tree.log_likelihood(random_df)
        # Mean log-likelihood of training data should be higher than random
        self.assertGreater(self.logp.mean(), random_logp.mean())


class TestRankRowsByChowLiu(unittest.TestCase):
    """Test rank_rows_by_chow_liu() produces expected output columns and scores."""

    @classmethod
    def setUpClass(cls):
        cls.df = make_correlated_survey(n_rows=300, seed=42)
        cls.ranked_df, cls.model = rank_rows_by_chow_liu(cls.df, alpha=1.0)

    def test_returns_dataframe_and_model(self):
        self.assertIsInstance(self.ranked_df, pd.DataFrame)
        self.assertIsInstance(self.model, CLTree)

    def test_scoring_columns_present(self):
        expected_cols = {"logp", "avg_logp", "gmean_prob", "rank_desc", "pct", "z"}
        self.assertTrue(expected_cols.issubset(set(self.ranked_df.columns)))

    def test_original_columns_preserved(self):
        for col in self.df.columns:
            self.assertIn(col, self.ranked_df.columns)

    def test_row_count_preserved(self):
        self.assertEqual(len(self.ranked_df), len(self.df))

    def test_rank_desc_is_valid_permutation(self):
        ranks = sorted(self.ranked_df["rank_desc"].tolist())
        self.assertEqual(ranks, list(range(1, len(self.df) + 1)))

    def test_pct_in_zero_one_range(self):
        self.assertTrue((self.ranked_df["pct"] >= 0).all())
        self.assertTrue((self.ranked_df["pct"] <= 1.0 + 1e-9).all())

    def test_gmean_prob_in_zero_one_range(self):
        self.assertTrue((self.ranked_df["gmean_prob"] >= 0).all())
        self.assertTrue((self.ranked_df["gmean_prob"] <= 1.0 + 1e-9).all())

    def test_z_scores_centered(self):
        z = self.ranked_df["z"]
        self.assertAlmostEqual(z.mean(), 0.0, places=3)

    def test_logp_and_avg_logp_consistent(self):
        n_cols = len(self.df.columns)
        np.testing.assert_allclose(
            self.ranked_df["avg_logp"],
            self.ranked_df["logp"] / n_cols,
            atol=1e-10,
        )


class TestChowLiuOutlierDetection(unittest.TestCase):
    """Test that CLTree can detect injected anomalous rows."""

    def test_injected_random_rows_score_lower(self):
        """Rows with broken correlations should have lower log-likelihood."""
        df = make_correlated_survey(n_rows=500, seed=42)

        # Inject 50 rows where correlations are broken
        rng = np.random.RandomState(123)
        outlier_rows = pd.DataFrame({
            "q1": rng.choice(["a", "b", "c"], size=50),
            "q2": rng.choice(["x", "y"], size=50),
            "q3": rng.choice(["low", "med", "high"], size=50),
            "q4": rng.choice(["yes", "no"], size=50),
        })
        combined = pd.concat([df, outlier_rows], ignore_index=True)

        # Fit on clean data only, score all rows
        tree = CLTree(alpha=1.0).fit(df)
        logp = tree.log_likelihood(combined)

        clean_mean = logp[:500].mean()
        outlier_mean = logp[500:].mean()

        # Clean rows (with correlations) should score higher on average
        self.assertGreater(clean_mean, outlier_mean)

    def test_error_column_compatible_with_evaluate(self):
        """Verify error = 1 - pct produces values compatible with evaluate_on_condition."""
        df = make_correlated_survey(n_rows=200, seed=42)
        ranked_df, _ = rank_rows_by_chow_liu(df, alpha=1.0)
        ranked_df["error"] = 1.0 - ranked_df["pct"]

        # Error should be in [0, 1]
        self.assertTrue((ranked_df["error"] >= 0).all())
        self.assertTrue((ranked_df["error"] <= 1.0 + 1e-9).all())

        # Higher error = more anomalous = lower logp
        sorted_by_error = ranked_df.sort_values("error", ascending=False)
        # Top row by error should have the lowest logp
        self.assertEqual(
            sorted_by_error.iloc[0]["rank_desc"],
            ranked_df["rank_desc"].max(),
        )

    def test_handles_missing_values(self):
        """CLTree should handle NaN values gracefully (as 'nan' category)."""
        df = make_correlated_survey(n_rows=100, seed=42)
        # Inject some NaN values
        df.iloc[0, 0] = np.nan
        df.iloc[5, 2] = np.nan

        tree = CLTree(alpha=1.0).fit(df)
        logp = tree.log_likelihood(df)

        self.assertEqual(len(logp), 100)
        self.assertFalse(np.any(np.isnan(logp)))
        self.assertFalse(np.any(np.isinf(logp)))

    def test_single_column_dataframe(self):
        """CLTree should work on a single-column DataFrame (degenerate tree)."""
        df = pd.DataFrame({"q1": ["a", "b", "c", "a", "b", "a"] * 20})
        tree = CLTree(alpha=1.0).fit(df)
        logp = tree.log_likelihood(df)

        self.assertEqual(len(logp), 120)
        self.assertEqual(len(tree.edges()), 0)  # no edges with 1 node
        self.assertFalse(np.any(np.isnan(logp)))
