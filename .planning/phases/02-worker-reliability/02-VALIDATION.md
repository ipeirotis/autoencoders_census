---
phase: 2
slug: worker-reliability
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-03
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pytest.ini |
| **Quick run command** | `pytest tests/test_worker_reliability.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_worker_reliability.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 02-01 | 1 | WORK-01,02,03 | unit | `python -m pytest tests/test_message_validation.py -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 02-01 | 1 | WORK-04 | integration | `python -m pytest tests/test_idempotency.py -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 02-01 | 1 | WORK-05,06 | integration | `python -m pytest tests/test_ack_extension.py -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02-02 | 1 | WORK-08 | unit | `python -m pytest tests/test_status_transitions.py -x` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02-02 | 1 | WORK-07 | integration | `python -m pytest tests/test_firestore_transactions.py -x` | ❌ W0 | ⬜ pending |
| 02-03-01 | 02-03 | 2 | WORK-09-14 | integration | `python -m pytest tests/test_csv_validation.py -x` | ❌ W0 | ⬜ pending |
| 02-03-02 | 02-03 | 2 | WORK-12 | unit | `npm test -- routes/jobs.test.ts` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_message_validation.py` — Pydantic message model tests
- [ ] `tests/test_idempotency.py` — Firestore idempotency tracking tests
- [ ] `tests/test_ack_extension.py` — Threading-based ack deadline extension tests
- [ ] `tests/test_status_transitions.py` — State machine validation tests
- [ ] `tests/test_firestore_transactions.py` — Transaction decorator tests
- [ ] `tests/test_csv_validation.py` — CSV encoding/structure/edge case tests
- [ ] `frontend/server/__tests__/routes/jobs.test.ts` — Express layer CSV validation tests

*Wave 0 creates test infrastructure per TDD approach before implementation begins.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vertex AI job submission | WORK-05 | Requires live GCP credentials and quota | 1. Run worker in vertex mode<br>2. Upload test CSV<br>3. Verify job appears in Vertex AI console<br>4. Confirm ack deadline extension logs appear |
| 10-15 min job completion | WORK-05, WORK-06 | Time-dependent, requires full training cycle | 1. Submit real CSV to trigger Vertex AI job<br>2. Monitor for 15 minutes<br>3. Verify no duplicate message delivery<br>4. Confirm status updates correctly |

*Automated tests will use mocks for Pub/Sub and Firestore. Manual verification required for end-to-end GCP integration.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-03
