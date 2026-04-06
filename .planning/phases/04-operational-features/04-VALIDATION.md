---
phase: 04
slug: operational-features
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-06
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (backend), vitest (frontend) |
| **Config file** | pytest.ini, vitest.config.ts |
| **Quick run command** | `pytest tests/ -v -k "export or lifecycle or contribution"` |
| **Full suite command** | `pytest tests/ -v && cd frontend && npm test` |
| **Estimated runtime** | ~15 seconds (quick), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -v -k "export or lifecycle or contribution"`
- **After every plan wave:** Run `pytest tests/ -v && cd frontend && npm test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

Tasks will be defined during planning. Examples:
- CSV export endpoint test: verify formula injection prevention
- GCS lifecycle rule test: verify configuration applied
- Vertex AI cancellation test: verify cancel() called
- Per-column contribution test: verify decomposition math

---

## Wave 0 Requirements

- [x] `tests/test_export.py` — stubs for OPS-01 through OPS-04 (CSV export)
- [x] `tests/test_lifecycle.py` — stubs for OPS-07, OPS-08, OPS-13 (GCS lifecycle)
- [x] `tests/test_cancellation.py` — stubs for OPS-05 through OPS-08 (job cancellation cleanup)
- [x] `frontend/client/components/__tests__/PerColumnScores.test.tsx` — stubs for OPS-09, OPS-10 (UI display)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GitHub PR workflow understanding | GH-01 through GH-05 | Documentation comprehension, not code | Maintainer reviews CONTRIBUTING.md and demonstrates understanding by creating sample PR |
| Expired job UI message | OPS-14 | Time-based (7-day wait impractical for CI) | Manually set job createdAt to 8 days ago in Firestore, verify UI shows "expired" message |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** Wave 0 complete (plan 04-00 creates test stubs)
