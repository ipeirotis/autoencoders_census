# Project State

## Current Position

**Phase:** Not started (defining requirements)
**Plan:** —
**Status:** Defining requirements
**Last activity:** 2026-03-24 — Milestone v1.0 started

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-24)

**Core value:** Researchers can confidently identify and remove low-quality survey responses by detecting multivariate patterns that traditional methods miss, ensuring their data analysis is based on genuine participant responses.

**Current focus:** Milestone v1.0 Production-Ready Web Platform

## Accumulated Context

<!-- Knowledge that persists across phases -->

### What Works
- CLI pipeline (train, evaluate, find_outliers) fully functional
- Web upload flow (React → Express → GCS → Pub/Sub → Worker)
- Vertex AI integration via CustomContainerTrainingJob
- Local processing mode (worker.py --mode=local) for development
- Tests passing for core pipeline
- GitHub Actions CI pipeline

### What Doesn't Work
- Frontend build (missing dependencies: lib/utils.ts, react-router-dom, serverless-http)
- Frontend runtime (no error boundary, polling loop issues, memory leak in CSV parser)
- Security (open CORS, path traversal, no auth, no rate limiting)
- Worker reliability (ack race condition, no message validation, hardcoded GCP IDs)
- Operational gaps (no progress feedback, no job cancellation, no CSV export)

### Technical Constraints
- Apple Silicon Macs need tensorflow-macos for local dev (not in requirements.txt)
- Vertex AI cold start takes 10-15 minutes (container initialization)
- Rule of 9 filter hardcoded (max 9 unique values) - configurable in future
- No authentication on API endpoints (private network only for now)

### User Expectations
- Upload CSV, get outlier scores back
- See progress during 10-15 min Vertex AI processing
- Download results as CSV
- Cancel jobs if needed
- Clear error messages when upload fails

---
*Last updated: 2026-03-24*
