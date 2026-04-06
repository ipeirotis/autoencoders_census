# Roadmap: AutoEncoder Outlier Detection Platform

**Milestone:** v1.0 Production-Ready Web Platform
**Created:** 2026-03-24
**Status:** Active

## Overview

This roadmap transforms the existing functional AutoEncoder platform into a production-ready web application. The 4 phases prioritize security blockers, then reliability, then UX polish. Every phase delivers observable capabilities that move closer to public deployment readiness.

## Phases

- [x] **Phase 1: Security Foundation** - Block deployment vulnerabilities before public access
- [x] **Phase 2: Worker Reliability** - Prevent race conditions and quota waste in async processing
- [ ] **Phase 3: Frontend Production** - Enable graceful error handling and user feedback for long jobs
- [ ] **Phase 4: Operational Features** - Add export, cleanup, and GitHub workflow capabilities

## Phase Details

### Phase 1: Security Foundation
**Goal**: Application can be safely deployed to public internet without critical security vulnerabilities.

**Depends on**: Nothing (first phase)

**Requirements**: SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, SEC-07, SEC-08, SEC-09, SEC-10, SEC-11, SEC-12, SEC-13, SEC-14, SEC-15, SEC-16 (16 requirements)

**Success Criteria** (what must be TRUE):
1. User must authenticate before uploading files or accessing job results
2. Rate limiting prevents abuse (upload floods, polling spam, download exhaustion)
3. File upload only accepts CSV files validated by magic bytes (not just extension)
4. User-provided filenames cannot escape storage directory or read arbitrary files
5. Malformed CSV inputs return clear error messages without exposing stack traces
6. CORS configuration blocks requests from domains outside approved frontend origin
7. Worker fails fast at startup if critical environment variables are missing

**Plans:** 7 plans

Plans:
- [x] 01-01-PLAN.md — Env validation, logging, error handling infrastructure (COMPLETE - 2026-03-29)
- [x] 01-02-PLAN.md — CORS whitelist and security headers (helmet) (COMPLETE - 2026-03-29)
- [x] 01-03-PLAN.md — Email/password authentication with Passport.js (COMPLETE - 2026-03-30)
- [x] 01-04-PLAN.md — Per-endpoint rate limiting (upload, poll, download) (COMPLETE - 2026-03-30)
- [x] 01-05-PLAN.md — Input validation and file security (CSV, path traversal) (COMPLETE - 2026-03-30)
- [x] 01-06-PLAN.md — Wire security stack into routes, integration tests (COMPLETE - 2026-04-02)
- [ ] 01-07-PLAN.md — Close verification gaps (legacy route, logging consistency)

---

### Phase 2: Worker Reliability
**Goal**: Async job processing handles duplicate messages, race conditions, and arbitrary CSV formats without wasting Vertex AI quota or corrupting job status.

**Depends on**: Phase 1 (relies on input validation patterns from security layer)

**Requirements**: WORK-01, WORK-02, WORK-03, WORK-04, WORK-05, WORK-06, WORK-07, WORK-08, WORK-09, WORK-10, WORK-11, WORK-12, WORK-13, WORK-14 (14 requirements)

**Success Criteria** (what must be TRUE):
1. Worker rejects Pub/Sub messages missing jobId, bucket, or file field
2. Duplicate Pub/Sub message delivery does not trigger duplicate Vertex AI jobs
3. Long-running jobs (10-15 min Vertex AI training) do not timeout and reprocess
4. Concurrent status updates do not create race conditions (job stuck in "training" when completed)
5. Worker processes CSV files with unicode characters, mixed types, mostly-missing values, and very wide datasets without crashing
6. Invalid CSV formats (encoding errors, inconsistent row lengths) return descriptive error messages to user

**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Message validation, idempotency, ack deadline extension (COMPLETE - 2026-04-05)
- [x] 02-02-PLAN.md — Transactional status updates and state machine validation (COMPLETE - 2026-04-05)
- [x] 02-03-PLAN.md — CSV validation with encoding detection and edge case handling (COMPLETE - 2026-04-05)

---

### Phase 3: Frontend Production
**Goal**: Frontend provides production-quality UX with error recovery, progress feedback, and job control for 10-15 minute async workflows.

**Depends on**: Phase 2 (progress tracking requires reliable Firestore status updates from worker)

**Requirements**: FE-01, FE-02, FE-03, FE-04, FE-05, FE-06, FE-07, FE-08, FE-09, FE-10, FE-11, FE-12, FE-13, FE-14, FE-15, FE-16, FE-17, FE-18, FE-19, FE-20, FE-21, FE-22 (22 requirements)

**Success Criteria** (what must be TRUE):
1. Component crashes display recovery UI instead of blank screen (error boundaries active)
2. User sees multi-stage progress indicator (Queued → Preprocessing → Training → Scoring) during job processing
3. User can cancel long-running job from UI (cancel button visible and functional)
4. Polling intervals stop when job completes or component unmounts (no memory leaks)
5. Application builds successfully with all dependencies installed (no missing lib/utils.ts, react-router-dom, serverless-http)
6. TypeScript strict mode catches type errors at compile time without unsafe assertions

**Plans**: 6 plans (split from original 4 to meet 200-line limit)

Plans:
- [ ] 03-01-PLAN.md — Build infrastructure and GCP client consolidation (Wave 1)
- [ ] 03-02-PLAN.md — React error boundaries for graceful error recovery (Wave 1)
- [ ] 03-03A-PLAN.md — Multi-stage progress tracking (Wave 2)
- [ ] 03-03B-PLAN.md — Job cancellation UI and API (Wave 2)
- [ ] 03-04A-PLAN.md — TypeScript strict mode (noImplicitAny) (Wave 1)
- [x] 03-04B-PLAN.md — Streaming CSV parser with Papa Parse and unified file validation (Wave 1) (COMPLETE - 2026-04-06)

---

### Phase 4: Operational Features
**Goal**: Users can export results, cancel jobs with resource cleanup, and maintainer understands GitHub PR workflow for ongoing collaboration.

**Depends on**: Phase 3 (export and cancellation features integrate with frontend UI)

**Requirements**: OPS-01, OPS-02, OPS-03, OPS-04, OPS-05, OPS-06, OPS-07, OPS-08, OPS-09, OPS-10, OPS-11, OPS-12, OPS-13, OPS-14, GH-01, GH-02, GH-03, GH-04, GH-05 (19 requirements)

**Success Criteria** (what must be TRUE):
1. User can download outlier results as CSV file without Excel formula injection risk
2. Canceled jobs clean up GCS files and cancel running Vertex AI jobs (not just Firestore flag)
3. Old uploaded files and results automatically delete after retention period (GCS lifecycle rules active)
4. User can see per-column outlier contribution scores in results UI (which survey questions were anomalous)
5. Maintainer understands branch strategy, commit conventions, and PR review process for collaborating with IliasTriant

**Plans**: TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Security Foundation | 6/7 | In Progress | - |
| 2. Worker Reliability | 3/3 | Complete | 2026-04-05 |
| 3. Frontend Production | 1/6 | In Progress | - |
| 4. Operational Features | 0/TBD | Not started | - |

---

## Coverage Validation

**Total v1.0 requirements**: 71
- Security Hardening (SEC): 16 requirements → Phase 1
- Worker Reliability (WORK): 14 requirements → Phase 2
- Frontend Production (FE): 22 requirements → Phase 3
- Operational Features (OPS): 14 requirements → Phase 4
- GitHub Best Practices (GH): 5 requirements → Phase 4

**Mapped to phases**: 71/71 (100%)
**Unmapped**: 0

---

*Roadmap created: 2026-03-24*
*Last updated: 2026-04-06*
