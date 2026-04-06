"""
Test stubs for CSV export functionality (Phase 4, Plans 04-01).
Tests will be implemented as features are built.
"""
import pytest


@pytest.mark.skip(reason="Pending plan 04-01 implementation")
def test_csv_export_endpoint_exists():
    """OPS-01: Verify /api/jobs/:id/export endpoint exists and requires auth"""
    pass


@pytest.mark.skip(reason="Pending plan 04-01 implementation")
def test_csv_export_formula_injection_protection():
    """OPS-02: Verify dangerous characters prefixed with single quote"""
    pass


@pytest.mark.skip(reason="Pending plan 04-01 implementation")
def test_csv_export_outlier_rows_only():
    """OPS-03: Verify export contains only outlier rows with scores"""
    pass


@pytest.mark.skip(reason="Pending plan 04-01 implementation")
def test_csv_export_content_disposition_header():
    """OPS-04: Verify response has Content-Disposition header for download"""
    pass
