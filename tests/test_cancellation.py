"""
Test stubs for job cancellation with resource cleanup (Phase 4, Plans 04-02).
Tests will be implemented as features are built.
"""
import pytest


@pytest.mark.skip(reason="Pending plan 04-02 implementation")
def test_cancel_job_deletes_gcs_files():
    """OPS-05: Verify canceled job deletes GCS uploaded file"""
    pass


@pytest.mark.skip(reason="Pending plan 04-02 implementation")
def test_cancel_job_calls_vertex_ai_cancel():
    """OPS-06: Verify canceled job calls Vertex AI jobs.cancel API"""
    pass


@pytest.mark.skip(reason="Pending plan 04-02 implementation")
def test_cancel_job_updates_firestore_status():
    """OPS-07: Verify canceled job updates Firestore status to 'canceled'"""
    pass


@pytest.mark.skip(reason="Pending plan 04-02 implementation")
def test_cancel_endpoint_requires_auth():
    """OPS-08: Verify DELETE /api/jobs/:id requires authentication"""
    pass
