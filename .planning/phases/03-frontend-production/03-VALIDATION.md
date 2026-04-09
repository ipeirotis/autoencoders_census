---
phase: 03
slug: frontend-production
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest (frontend), pytest (backend) |
| **Config file** | frontend/vitest.config.ts, pytest.ini |
| **Quick run command** | `cd frontend && npm run test:unit -- --run` |
| **Full suite command** | `cd frontend && npm run test:unit -- --run && cd .. && python -m pytest tests/` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd frontend && npm run test:unit -- --run`
- **After every plan wave:** Run full suite (frontend + backend)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | FE-11 | unit | `npm test lib/utils.spec.ts` | ✅ | ⬜ pending |
| 03-01-02 | 01 | 1 | FE-12 | integration | `npm run build` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | FE-01 | unit | `npm test ErrorBoundary.spec.tsx` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 2 | FE-04 | unit | `npm test ProgressPage.spec.tsx` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 2 | FE-08 | unit | `npm test useJobPolling.spec.ts` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 3 | FE-16 | integration | `npm run type-check` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `frontend/client/components/ErrorBoundary.spec.tsx` — unit tests for error boundary component
- [ ] `frontend/client/pages/ProgressPage.spec.tsx` — tests for progress indicator UI
- [ ] `frontend/client/hooks/useJobPolling.spec.ts` — tests for polling lifecycle
- [ ] `frontend/vitest.config.ts` — vitest configuration (may already exist)

*Wave 0 plan creates test stubs before implementation plans execute.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Progress bar visual accuracy | FE-04 | Visual assertion | Load /job/:id page, verify step badges render correctly with current step highlighted |
| Error boundary recovery flow | FE-03 | User interaction | Trigger component crash, verify "Reload page" button works and restores app |
| Polling stops on unmount | FE-08 | Memory profiling | Navigate away from progress page, verify Firestore queries stop |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
