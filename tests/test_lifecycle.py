"""
Test stubs for GCS lifecycle rules (Phase 4, Plans 04-03).
Tests will be implemented as features are built.
"""
import pytest


@pytest.mark.skip(reason="Pending plan 04-03 implementation")
def test_gcs_lifecycle_rule_exists():
    """OPS-07: Verify GCS bucket has lifecycle rule configured"""
    pass


@pytest.mark.skip(reason="Pending plan 04-03 implementation")
def test_gcs_lifecycle_rule_7_day_retention():
    """OPS-08: Verify lifecycle rule deletes files after 7 days"""
    pass


@pytest.mark.skip(reason="Pending plan 04-03 implementation")
def test_expired_job_ui_hides_download_button():
    """OPS-13: Verify download button hidden for expired jobs"""
    pass
