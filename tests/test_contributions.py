"""
Tests for per-column contribution score computation.

Tests the compute_per_column_contributions function that decomposes
reconstruction loss into per-attribute percentages for outlier explanation.
"""

import numpy as np
from evaluate.outliers import compute_per_column_contributions


class TestPerColumnContributions:
    """Test suite for contribution score computation."""

    def test_returns_list_of_tuples(self):
        """Test that function returns list of (column_name, percentage) tuples."""
        # Simple 2-column case: each column has 3 categories (one-hot encoded)
        data = np.array([[1, 0, 0, 0, 1, 0]])  # 1 row, 6 features total
        predictions = np.array([[0.7, 0.2, 0.1, 0.1, 0.7, 0.2]])
        attr_cardinalities = [3, 3]
        column_names = ['col1', 'col2']

        result = compute_per_column_contributions(
            data, predictions, attr_cardinalities, column_names
        )

        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 2, "Should have 2 contributions (one per column)"
        assert all(isinstance(item, tuple) for item in result), "Each item should be a tuple"
        assert all(len(item) == 2 for item in result), "Each tuple should have 2 elements"
        assert all(isinstance(item[0], str) for item in result), "First element should be column name"
        assert all(isinstance(item[1], float) for item in result), "Second element should be percentage"

    def test_contributions_sum_to_100(self):
        """Test that contributions sum to approximately 100% (within 0.5% tolerance)."""
        data = np.array([[1, 0, 0, 0, 1, 0]])
        predictions = np.array([[0.7, 0.2, 0.1, 0.1, 0.7, 0.2]])
        attr_cardinalities = [3, 3]
        column_names = ['col1', 'col2']

        result = compute_per_column_contributions(
            data, predictions, attr_cardinalities, column_names
        )

        total = sum(pct for _, pct in result)
        assert 99.5 <= total <= 100.5, f"Contributions should sum to ~100%, got {total}"

    def test_sorted_descending_by_percentage(self):
        """Test that contributions are sorted descending by percentage."""
        # Create data where col2 has higher loss than col1
        data = np.array([[1, 0, 0, 0, 1, 0]])
        # col2 has worse prediction (0.1 vs true 0-1-0)
        predictions = np.array([[0.9, 0.05, 0.05, 0.1, 0.1, 0.8]])
        attr_cardinalities = [3, 3]
        column_names = ['col1', 'col2']

        result = compute_per_column_contributions(
            data, predictions, attr_cardinalities, column_names
        )

        # Check descending order
        percentages = [pct for _, pct in result]
        assert percentages == sorted(percentages, reverse=True), "Contributions should be sorted descending"

    def test_handles_single_row(self):
        """Test that function handles single-row input (shape [1, features])."""
        data = np.array([[1, 0, 0, 0, 1, 0]])  # Single row
        predictions = np.array([[0.7, 0.2, 0.1, 0.1, 0.7, 0.2]])
        attr_cardinalities = [3, 3]
        column_names = ['col1', 'col2']

        result = compute_per_column_contributions(
            data, predictions, attr_cardinalities, column_names
        )

        assert len(result) == 2, "Should handle single row"
        total = sum(pct for _, pct in result)
        assert 99.5 <= total <= 100.5, "Single row contributions should sum to ~100%"

    def test_handles_batch_input(self):
        """Test that function handles batch input (shape [N, features])."""
        # 3 rows
        data = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1]
        ])
        predictions = np.array([
            [0.7, 0.2, 0.1, 0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2, 0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7, 0.2, 0.1, 0.7]
        ])
        attr_cardinalities = [3, 3]
        column_names = ['col1', 'col2']

        result = compute_per_column_contributions(
            data, predictions, attr_cardinalities, column_names
        )

        assert len(result) == 2, "Should handle batch input"
        total = sum(pct for _, pct in result)
        assert 99.5 <= total <= 100.5, "Batch contributions should sum to ~100%"

    def test_zero_total_loss_returns_equal_contributions(self):
        """Test that zero total loss returns equal contributions (fallback behavior)."""
        # Perfect predictions (zero loss)
        data = np.array([[1, 0, 0, 0, 1, 0]])
        predictions = np.array([[1, 0, 0, 0, 1, 0]])  # Exact match
        attr_cardinalities = [3, 3]
        column_names = ['col1', 'col2']

        result = compute_per_column_contributions(
            data, predictions, attr_cardinalities, column_names
        )

        # Should fallback to equal contribution
        percentages = [pct for _, pct in result]
        assert all(abs(pct - 50.0) < 0.1 for pct in percentages), "Zero loss should give equal contributions"
