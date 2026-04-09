---
phase: 01
slug: security-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Jest 29.x + Supertest 6.x (Express API) / pytest 7.x (Python worker) |
| **Config file** | frontend/jest.config.js (Wave 0 installs) / pytest.ini (exists) |
| **Quick run command** | `npm test -- --testPathPattern={changed-file}` |
| **Full suite command** | `npm test && python -m pytest tests/test_env_validation.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `npm test -- --testPathPattern={changed-file}` (< 10s)
- **After every plan wave:** Run `npm test` (< 30s)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | - | setup | `npm test -- --version` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | SEC-01 | integration | `npm test -- server/__tests__/auth.test.ts` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | SEC-01 | integration | `npm test -- server/__tests__/auth.test.ts` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 1 | SEC-02 | integration | `npm test -- server/__tests__/cors.test.ts` | ❌ W0 | ⬜ pending |
| 01-04-01 | 04 | 1 | SEC-03, SEC-04, SEC-05 | integration | `npm test -- server/__tests__/rateLimits.test.ts` | ❌ W0 | ⬜ pending |
| 01-04-02 | 04 | 1 | SEC-15 | integration | `npm test -- server/__tests__/rateLimits.test.ts` | ❌ W0 | ⬜ pending |
| 01-05-01 | 05 | 2 | SEC-06, SEC-07 | integration | `npm test -- server/__tests__/upload.test.ts` | ❌ W0 | ⬜ pending |
| 01-05-02 | 05 | 2 | SEC-08, SEC-09 | unit | `npm test -- server/__tests__/fileHandling.test.ts` | ❌ W0 | ⬜ pending |
| 01-06-01 | 06 | 2 | SEC-10, SEC-11 | integration | `npm test -- server/__tests__/validation.test.ts` | ❌ W0 | ⬜ pending |
| 01-07-01 | 07 | 2 | SEC-13 | integration | `npm test -- server/__tests__/errorHandler.test.ts` | ❌ W0 | ⬜ pending |
| 01-07-02 | 07 | 2 | SEC-14 | integration | `npm test -- server/__tests__/security.test.ts` | ❌ W0 | ⬜ pending |
| 01-08-01 | 08 | 1 | SEC-12, SEC-16 | unit | `python -m pytest tests/test_env_validation.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `frontend/jest.config.js` — Jest configuration for Node.js/TypeScript
- [ ] `frontend/server/__tests__/setup.ts` — Test database cleanup, mock Firestore/GCS
- [ ] `npm install --save-dev jest @types/jest supertest @types/supertest ts-jest` — Jest + Supertest installation
- [ ] `tests/test_env_validation.py` — Python worker env validation tests

*Wave 0 must complete before any other wave can run tests.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Session persists after server restart | SEC-01 | Requires Firestore state verification | 1. Login 2. Restart server 3. Verify session cookie still works |
| CORS blocks Postman with fake Origin | SEC-02 | Postman behavior varies by version | 1. Set Origin: http://evil.com in Postman 2. Verify 403 response |

*All other behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
