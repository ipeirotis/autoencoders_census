# AutoEncoder Outlier Detection Platform

## What This Is

A full-stack ML platform for detecting outliers in survey and tabular data using autoencoders. Users upload CSV files through a web interface and receive outlier analysis - identifying inattentive or mischievous survey respondents based on reconstruction error. The platform supports both a CLI for researchers and a web UI for casual users, with cloud-based training via Vertex AI.

## Core Value

Researchers can confidently identify and remove low-quality survey responses by detecting multivariate patterns that traditional methods miss, ensuring their data analysis is based on genuine participant responses.

## Current Milestone: v1.0 Production-Ready Web Platform

**Goal:** Harden the web application (frontend, backend, worker) to production quality with proper security, error handling, and operational reliability.

**Target features:**
- Robust CSV upload handling with validation and error reporting
- API authentication and security hardening (CORS, rate limiting, path traversal fixes)
- Worker reliability improvements (message validation, ack handling)
- Frontend build and runtime fixes (missing dependencies, error boundaries, proper polling)
- Operational polish (progress indicators, CSV export, job cancellation)
- GitHub workflow best practices (PR process, code review patterns)

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ **CLI Pipeline** - Train/evaluate/find_outliers commands work for built-in datasets — existing
- ✓ **Web Upload** - Users can upload CSV via React UI → Express → GCS — existing (v0.1)
- ✓ **Vertex AI Integration** - Worker dispatches training jobs to Vertex AI — existing (v0.1)
- ✓ **Results Display** - Frontend shows outlier scores and dropped columns — existing (v0.1)
- ✓ **Rule of 9 Filtering** - Automatic column selection for categorical data — existing
- ✓ **Local Processing Mode** - Worker can process uploads locally without Vertex AI — existing (worker.py --mode=local)
- ✓ **Basic Tests** - Core pipeline has passing integration tests — existing (v0.1)
- ✓ **CI Pipeline** - GitHub Actions runs tests on push/PR — existing (v0.1)

### Active

<!-- Current scope. Building toward these. -->

**Security & Validation (Section 2):**
- [ ] Robust handling of arbitrary CSV uploads (mixed types, unicode, wide datasets, mostly-missing)
- [ ] Input validation with clear error messages (malformed CSV, encoding issues, empty files)
- [ ] API authentication (minimum API key or session-based auth)
- [ ] CORS restrictions to actual frontend domain
- [ ] Path traversal protection (sanitize user-provided filenames)
- [ ] File type validation (restrict to .csv)
- [ ] Rate limiting on upload and polling endpoints
- [ ] Signed URL endpoint authentication
- [ ] Job ID validation before Firestore access
- [ ] Worker message field validation (jobId, bucket, file required)
- [ ] Worker ack deadline handling (prevent duplicate processing)
- [ ] Environment variable validation at worker startup
- [ ] Remove hardcoded GCP identifiers (use env vars)

**Frontend Production Readiness (Section 4):**
- [ ] Display per-column outlier contribution scores
- [ ] Progress indicator for long-running jobs (stage tracking in Firestore)
- [ ] CSV export button for results
- [ ] Job cancellation from UI
- [ ] Fix missing dependencies (lib/utils.ts, react-router-dom, serverless-http)
- [ ] Add missing npm scripts (build:client, dev:server)
- [ ] React error boundary for graceful failure
- [ ] Fix CSV parser memory issue (streaming for preview)
- [ ] Fix polling useEffect dependency on toast
- [ ] File type validation in click-upload path
- [ ] Enable TypeScript strict mode
- [ ] Deduplicate GCP client instances
- [ ] Resolve duplicate job-status routes
- [ ] Fix port mismatch (server vs docs)

**GitHub Best Practices:**
- [ ] Learn PR workflow (branch strategy, commit messages, code review)
- [ ] Understand collaborator patterns (how IliasTriant structures PRs)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- **Section 1 (Core Pipeline Bugs)** — Defer to v1.1: Data loading format inconsistencies, test fixes, CLI command bugs are important but don't block web UI production use. Users will primarily use the web interface.
- **Section 3 (Model Quality)** — Defer to v1.2: Per-column scores, benchmarking, numerical stability are valuable but not blockers for production deployment. Current model works adequately.
- **Section 8 (Missing Data Handling)** — Defer to v2.0: Masked loss approach is a significant research/architecture change. Current fillna("missing") strategy works for v1.
- **Section 9 (Strategic Concerns)** — Ongoing research: Rule of 9 limitations, baseline comparisons, evaluation methodology are long-term considerations, not v1 blockers.
- **Section 10 (Chow-Liu Tree)** — Defer to v2.0: Alternative outlier method exists in code but not integrated. Future enhancement.
- **Containerization (6.3)** — Defer: docker-compose setup is nice-to-have, not required for initial production deployment.
- **Mobile App** — Not building native apps; web-first approach for v1 and beyond.
- **Real-time Processing** — Async job queue is the architecture; no plans for synchronous/streaming results.

## Context

**Technical Environment:**
- Python 3.10+, TensorFlow 2.15.1, Keras
- React 18, Vite, TypeScript, Tailwind CSS, shadcn/ui
- Express.js API server
- Google Cloud: Storage, Pub/Sub, Firestore, Vertex AI
- GitHub Actions for CI

**Current State:**
- Codebase has functional CLI and web UI (v0.1 shipped)
- Comprehensive TASKS.md identifying 60+ production gaps
- Tests passing (sections 5.1, 5.2, 5.4 marked DONE)
- Security issues documented but not fixed
- Frontend has build blockers preventing deployment
- Worker has reliability issues (ack race, missing validation)

**Known Issues to Address:**
- Frontend missing critical dependencies (lib/utils.ts, react-router-dom)
- Security vulnerabilities (open CORS, path traversal, no rate limiting)
- Worker reliability (ack race, no field validation, hardcoded GCP IDs)
- Frontend runtime issues (no error boundary, polling loop bugs)
- Operational gaps (no progress tracking, no job cancellation)

**User Research:**
- Target users: Survey researchers, data scientists working with tabular data
- Primary workflow: Upload CSV → Wait 10-15 min (Vertex AI) → Download outlier scores
- Pain points: No progress feedback, can't cancel jobs, unclear errors, build doesn't work

## Constraints

- **Cloud Platform**: Google Cloud Platform — already integrated, all infrastructure exists
- **Timeline**: Production-ready for demo/initial users — no hard deadline but quality over speed
- **Compatibility**: Must maintain backward compatibility with existing CLI commands and built-in datasets
- **Apple Silicon**: Local dev must work on M1/M2/M3 Macs (tensorflow-macos for local, standard TensorFlow in container)
- **Security**: No deployment to public internet until authentication and security hardening complete
- **Testing**: All changes must have tests; CI must pass before merge

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Focus on Web UI production readiness first | CLI works, web UI has most gaps; users prefer web interface | — Pending |
| Defer core pipeline refactoring (Section 1) | Don't block web production on backend cleanup; users won't hit these via UI | — Pending |
| Out of scope: missing data masking (Section 8) | Architectural change requiring research; current approach adequate for v1 | — Pending |
| Require all security fixes before any public deployment | Open CORS, path traversal, no auth = critical vulnerabilities | — Pending |
| Learn GitHub best practices as part of this milestone | Collaborate effectively with IliasTriant; PRs are part of production workflow | — Pending |

---
*Last updated: 2026-03-24 after milestone v1.0 initialization*
