---
phase: 2
slug: worker-reliability
status: draft
nyquist_compliant: false
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
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*This table will be populated during planning when task IDs are assigned.*

---

## Wave 0 Requirements

- [ ] `tests/test_worker_reliability.py` — test stubs for WORK-01 through WORK-14
- [ ] pytest configuration validated — ensure pytest.ini exists and is configured

*Wave 0 creates test infrastructure before implementation begins.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Vertex AI job submission | WORK-05 | Requires live GCP credentials and quota | 1. Run worker in vertex mode<br>2. Upload test CSV<br>3. Verify job appears in Vertex AI console<br>4. Confirm ack deadline extension logs appear |
| 10-15 min job completion | WORK-05, WORK-06 | Time-dependent, requires full training cycle | 1. Submit real CSV to trigger Vertex AI job<br>2. Monitor for 15 minutes<br>3. Verify no duplicate message delivery<br>4. Confirm status updates correctly |

*Automated tests will use mocks for Pub/Sub and Firestore. Manual verification required for end-to-end GCP integration.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
