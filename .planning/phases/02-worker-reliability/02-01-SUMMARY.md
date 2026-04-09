---
phase: 02-worker-reliability
plan: 01
subsystem: worker
tags: [pubsub, firestore, pydantic, threading, idempotency, message-validation]

# Dependency graph
requires:
  - phase: 01-security-foundation
    provides: Firestore collections, GCS bucket, environment validation patterns
provides:
  - Pydantic message validation for Pub/Sub payloads
  - Idempotent message processing using Firestore transactions
  - Ack deadline extension for long-running jobs (10-15 min)
  - Delayed message acknowledgment (ack only after processing completes)
affects: [02-02-status-updates, 02-03-csv-validation, worker-monitoring]

# Tech tracking
tech-stack:
  added: [pydantic (already in requirements), threading.Timer]
  patterns: [Firestore transactional idempotency, periodic ack extension, delayed ack]

key-files:
  created:
    - tests/test_message_validation.py
    - tests/test_idempotency.py
    - tests/test_ack_extension.py
  modified:
    - worker.py

key-decisions:
  - "Use Pydantic for message validation instead of manual dict checks - provides clear error messages and type safety"
  - "Use Firestore transactions for idempotency to handle race conditions when multiple workers process same message"
  - "Use threading.Timer for ack extension instead of background thread - simpler cleanup and cancellation"
  - "Extend deadline by interval+10 seconds (70s total for 60s interval) to provide buffer before timeout"
  - "Move message.ack() to AFTER processing completes - prevents premature ack and loss of work on failure"

patterns-established:
  - "Pattern 1: Message validation with Pydantic BaseModel before processing - raises ValueError with field-specific errors"
  - "Pattern 2: Idempotency via Firestore transactional document check - prevents duplicate processing from Pub/Sub redelivery"
  - "Pattern 3: AckExtender class with start/stop methods - periodic extension every 60 seconds, cleanup in finally block"
  - "Pattern 4: Delayed acknowledgment - ack() called only after successful completion, not at start of processing"

requirements-completed: [WORK-01, WORK-02, WORK-03, WORK-04, WORK-05, WORK-06]

# Metrics
duration: 3min
completed: 2026-04-05
---

# Phase 02 Plan 01: Worker Reliability Foundation Summary

**Pub/Sub message processing hardened with Pydantic validation, Firestore-backed idempotency, and threading.Timer ack extension for 10-15 minute jobs**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-05T22:20:43Z
- **Completed:** 2026-04-05T22:23:16Z
- **Tasks:** 3
- **Files modified:** 4 (worker.py + 3 test files)

## Accomplishments
- Worker rejects malformed Pub/Sub messages with clear error messages (missing jobId, bucket, or file fields)
- Duplicate message delivery does not trigger duplicate Vertex AI jobs (Firestore transaction-based deduplication)
- Long-running jobs extend ack deadline every 60 seconds to prevent timeout and redelivery
- Message acknowledged only after processing completes successfully (not before, preventing work loss on failure)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Pydantic message validation** - `fe2415f` (test - includes implementation)
2. **Task 2: Add idempotent processing with Firestore** - `34d315c` (test), `3f96c3d` (feat)
3. **Task 3: Add ack deadline extension for long jobs** - `b442d0c` (feat)

_Note: Tasks 1 and 2 were committed by a previous agent. Task 1 combined test and implementation in one commit. Task 3 follows TDD with separate test file already created._

## Files Created/Modified

Created:
- `tests/test_message_validation.py` - Tests for Pydantic validation (missing fields, valid message, callback nacking)
- `tests/test_idempotency.py` - Tests for Firestore idempotency (duplicate detection, transaction race conditions)
- `tests/test_ack_extension.py` - Tests for periodic ack extension (timer scheduling, cleanup, error handling)

Modified:
- `worker.py` - Added PubSubMessage Pydantic model, validate_message(), check_idempotency(), AckExtender class, updated callback() and processing functions

## Decisions Made

1. **Pydantic for validation** - Clearer error messages than manual dict checks, automatic type coercion, min_length enforcement
2. **Firestore transactions for idempotency** - Atomic read-modify-write prevents race conditions when two workers check same message simultaneously
3. **threading.Timer for extension** - Simpler than dedicated background thread, easier cleanup with cancel() method
4. **70-second extension for 60-second interval** - 10-second buffer ensures deadline never expires between extensions
5. **Delayed ack after completion** - Moved message.ack() from start of processing to after completion, ensures work not lost if worker crashes mid-processing

## Deviations from Plan

None - plan executed exactly as written. All three tasks implemented according to specifications with no unplanned work.

## Issues Encountered

None - implementation straightforward. Tests passed on first run after adding AckExtender class.

## User Setup Required

None - no external service configuration required. Uses existing Firestore database and Pub/Sub subscription from Phase 01.

## Next Phase Readiness

**Ready for Plan 02-02 (Status Updates)**: Worker now has robust message handling foundation. Status update implementation can safely assume messages are validated, deduplicated, and won't timeout.

**Blockers:** None

**Notes:**
- All 14 tests passing (5 validation + 4 idempotency + 5 ack extension)
- Ack extension tested with fast intervals (0.1s) to avoid slow tests, production uses 60s
- Idempotency uses 7-day TTL hint in Firestore (manual cleanup required, no auto-TTL in Firestore)

## Self-Check: PASSED

All files and commits verified:
- ✓ tests/test_message_validation.py exists
- ✓ tests/test_idempotency.py exists
- ✓ tests/test_ack_extension.py exists
- ✓ Commit b442d0c exists (Task 3)
- ✓ Commit 3f96c3d exists (Task 2 feat)
- ✓ Commit fe2415f exists (Task 1 test+impl)

---
*Phase: 02-worker-reliability*
*Plan: 01*
*Completed: 2026-04-05*
