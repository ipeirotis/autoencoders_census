"""
Test status transition state machine validation.

Tests enforce business rules:
- Valid forward transitions (queued → processing → training → scoring → complete)
- Backward transitions rejected (complete → processing)
- Terminal states cannot transition
- Error and canceled states reachable from any state
"""

from worker import JobStatus, is_valid_transition


class TestStatusTransitions:
    """Test the job status state machine."""

    def test_valid_transition_queued_to_processing(self):
        """Test 1: is_valid_transition() allows queued → processing"""
        assert is_valid_transition(JobStatus.QUEUED, JobStatus.PROCESSING) is True

    def test_valid_transition_full_happy_path(self):
        """Test 2: is_valid_transition() allows processing → training → scoring → complete"""
        # processing → training
        assert is_valid_transition(JobStatus.PROCESSING, JobStatus.TRAINING) is True
        # training → scoring
        assert is_valid_transition(JobStatus.TRAINING, JobStatus.SCORING) is True
        # scoring → complete
        assert is_valid_transition(JobStatus.SCORING, JobStatus.COMPLETE) is True

    def test_invalid_transition_complete_to_processing(self):
        """Test 3: is_valid_transition() rejects complete → processing (backward)"""
        assert is_valid_transition(JobStatus.COMPLETE, JobStatus.PROCESSING) is False

    def test_invalid_transition_training_to_queued(self):
        """Test 4: is_valid_transition() rejects training → queued (backward)"""
        assert is_valid_transition(JobStatus.TRAINING, JobStatus.QUEUED) is False

    def test_valid_transition_any_to_error(self):
        """Test 5: is_valid_transition() allows any state → error (error handling)"""
        assert is_valid_transition(JobStatus.QUEUED, JobStatus.ERROR) is True
        assert is_valid_transition(JobStatus.PROCESSING, JobStatus.ERROR) is True
        assert is_valid_transition(JobStatus.TRAINING, JobStatus.ERROR) is True
        assert is_valid_transition(JobStatus.SCORING, JobStatus.ERROR) is True

    def test_valid_transition_any_to_canceled(self):
        """Test 6: is_valid_transition() allows any state → canceled (cancellation)"""
        assert is_valid_transition(JobStatus.QUEUED, JobStatus.CANCELED) is True
        assert is_valid_transition(JobStatus.PROCESSING, JobStatus.CANCELED) is True
        assert is_valid_transition(JobStatus.TRAINING, JobStatus.CANCELED) is True
        assert is_valid_transition(JobStatus.SCORING, JobStatus.CANCELED) is True

    def test_terminal_states_reject_all_transitions(self):
        """Test 7: Terminal states (complete, error, canceled) reject all transitions"""
        # complete is terminal
        assert is_valid_transition(JobStatus.COMPLETE, JobStatus.PROCESSING) is False
        assert is_valid_transition(JobStatus.COMPLETE, JobStatus.ERROR) is False
        assert is_valid_transition(JobStatus.COMPLETE, JobStatus.CANCELED) is False

        # error is terminal
        assert is_valid_transition(JobStatus.ERROR, JobStatus.PROCESSING) is False
        assert is_valid_transition(JobStatus.ERROR, JobStatus.COMPLETE) is False
        assert is_valid_transition(JobStatus.ERROR, JobStatus.CANCELED) is False

        # canceled is terminal
        assert is_valid_transition(JobStatus.CANCELED, JobStatus.PROCESSING) is False
        assert is_valid_transition(JobStatus.CANCELED, JobStatus.COMPLETE) is False
        assert is_valid_transition(JobStatus.CANCELED, JobStatus.ERROR) is False

    def test_first_status_update_must_be_queued(self):
        """Test that first status update (from None) must be QUEUED"""
        assert is_valid_transition(None, JobStatus.QUEUED) is True
        assert is_valid_transition(None, JobStatus.PROCESSING) is False
        assert is_valid_transition(None, JobStatus.COMPLETE) is False

    def test_unknown_status_values_rejected(self):
        """Test that unknown status string values are rejected"""
        assert is_valid_transition("queued", "invalid_status") is False
        assert is_valid_transition("unknown_status", "processing") is False
