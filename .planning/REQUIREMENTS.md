# Requirements: AutoEncoder Outlier Detection Platform

**Defined:** 2026-03-24
**Core Value:** Researchers can confidently identify and remove low-quality survey responses by detecting multivariate patterns that traditional methods miss, ensuring their data analysis is based on genuine participant responses.

## v1.0 Requirements

Requirements for production-ready web platform. Each maps to roadmap phases.

### Security Hardening

- [x] **SEC-01**: API implements authentication layer (session-based for UI, API keys for service-to-server)
- [x] **SEC-02**: CORS restricts requests to actual frontend domain (no wildcard on authenticated endpoints)
- [x] **SEC-03**: Rate limiting applied to upload endpoint (5 uploads per 15 minutes per user)
- [x] **SEC-04**: Rate limiting applied to polling endpoint (60 status checks per minute per user)
- [x] **SEC-05**: Rate limiting applied to download endpoint (10 downloads per hour per user)
- [x] **SEC-06**: File type validation checks magic bytes (not just extension)
- [x] **SEC-07**: File type validation restricts to .csv MIME types only
- [x] **SEC-08**: Path traversal protection uses UUID filenames (discards user-provided names)
- [x] **SEC-09**: Path traversal protection canonicalizes all file paths
- [x] **SEC-10**: Input validation sanitizes all user inputs with express-validator
- [x] **SEC-11**: Input validation provides clear error messages for invalid inputs
- [x] **SEC-12**: Environment variable validation runs at worker startup (fails fast if missing)
- [x] **SEC-13**: Error handling never exposes stack traces in production responses
- [x] **SEC-14**: Helmet middleware applies security headers (CSP, X-Frame-Options, etc.)
- [x] **SEC-15**: Rate limiting uses version 8.0.2+ to avoid CVE-2026-30827 (IPv6 bypass)
- [x] **SEC-16**: Hardcoded GCP identifiers replaced with environment variables

### Worker Reliability

- [x] **WORK-01**: Pub/Sub message validation requires jobId field
- [x] **WORK-02**: Pub/Sub message validation requires bucket field
- [x] **WORK-03**: Pub/Sub message validation requires file field
- [x] **WORK-04**: Idempotent processing tracks processed message IDs in Firestore
- [x] **WORK-05**: Ack deadline extended during long-running Vertex AI jobs (10-15 min)
- [x] **WORK-06**: Message acknowledged only after processing completes successfully
- [x] **WORK-07**: Firestore status updates use transactions (prevent race conditions)
- [x] **WORK-08**: Status transition validation prevents backward state changes
- [x] **WORK-09**: CSV streaming validates file encoding before processing
- [x] **WORK-10**: CSV streaming validates file structure (headers, row consistency)
- [x] **WORK-11**: CSV streaming enforces size limits with clear error messages
- [x] **WORK-12**: CSV validation occurs at both Express layer and Worker layer (defense-in-depth)
- [x] **WORK-13**: Worker handles arbitrary CSV formats (mixed types, unicode, special chars)
- [x] **WORK-14**: Worker handles edge cases (mostly-missing values, very wide datasets)

### Frontend Production

- [x] **FE-01**: React error boundary wraps App component (catches render errors)
- [x] **FE-02**: React error boundary wraps high-risk components (Preview, Results)
- [x] **FE-03**: Error boundaries display recovery UI (not blank screen)
- [x] **FE-04**: Progress indicator shows multi-stage status (Queued → Preprocessing → Training → Scoring)
- [x] **FE-05**: Progress indicator displays percent complete for each stage
- [x] **FE-06**: Job cancellation UI provides cancel button on job status page
- [x] **FE-07**: Job cancellation confirms with user before canceling
- [x] **FE-08**: Polling interval cleanup prevents memory leaks on unmount
- [x] **FE-09**: Polling stops when job completes (completed/failed/canceled states)
- [x] **FE-10**: Polling useEffect dependencies fixed (no stale closures with toast)
- [x] **FE-11**: Missing dependency lib/utils.ts created (cn utility for shadcn/ui)
- [x] **FE-12**: Missing dependency react-router-dom added to package.json
- [x] **FE-13**: Missing dependency serverless-http added to package.json
- [x] **FE-14**: Missing npm script build:client added to package.json
- [x] **FE-15**: Missing npm script dev:server added to package.json
- [x] **FE-16**: TypeScript strict mode enabled (incremental: noImplicitAny → strictNullChecks → strict)
- [x] **FE-17**: TypeScript strict mode violations fixed without type assertions
- [x] **FE-18**: GCP client instances deduplicated (single Storage/Firestore/PubSub instance)
- [x] **FE-19**: Duplicate job-status routes resolved (index.ts vs routes/jobs.ts)
- [x] **FE-20**: Port mismatch fixed (server uses documented port)
- [x] **FE-21**: CSV parser uses streaming for preview (prevents memory crash on large files)
- [x] **FE-22**: File type validation added to click-upload path (not just drag-and-drop)

### Operational Features

- [x] **OPS-01**: User can export outlier results as CSV file
- [x] **OPS-02**: CSV export prevents formula injection (sanitizes =, +, -, @, \t, \r characters)
- [x] **OPS-03**: CSV export includes proper Content-Disposition headers
- [x] **OPS-04**: Job cancellation deletes GCS files for canceled jobs
- [x] **OPS-05**: Job cancellation cancels Vertex AI job if running
- [x] **OPS-06**: Job cancellation updates Firestore status to "canceled"
- [x] **OPS-07**: GCS lifecycle rules delete old uploaded files (30-day retention)
- [x] **OPS-08**: GCS lifecycle rules delete old result files (90-day retention)
- [x] **OPS-09**: User can see per-column outlier contribution scores in results
- [x] **OPS-10**: Per-column scores show which survey questions were anomalous
- [ ] **OPS-11**: User can download failed-rows CSV with specific error descriptions
- [ ] **OPS-12**: Row-level validation errors indicate encoding issues, missing values, schema mismatches
- [x] **OPS-13**: Signed URLs generated on-demand (1-hour expiration, not 7-day)
- [x] **OPS-14**: Progress tracking writes stage updates to Firestore throughout processing

### GitHub Best Practices

- [x] **GH-01**: Understand PR workflow (branch strategy, naming conventions)
- [x] **GH-02**: Understand commit message conventions (used in this repository)
- [x] **GH-03**: Understand code review process (how to request reviews, address feedback)
- [x] **GH-04**: Analyze IliasTriant's PR patterns (structure, descriptions, commits)
- [x] **GH-05**: Practice creating well-structured PRs for v1.0 features

## v2.0 Requirements

Deferred to future releases. Tracked but not in current roadmap.

### Missing Data Handling

- **DATA-01**: Mask missing attributes out of loss function (train only on observed values)
- **DATA-02**: Use decoder predictions on masked positions for imputation recommendations
- **DATA-03**: Add binary missing indicator columns (alongside all-zeros encoding)
- **DATA-04**: Compute per-attribute mask tensor throughout pipeline
- **DATA-05**: Update outlier scoring to use only observed attributes
- **DATA-06**: Add missingness_fraction metric (fraction of missing attributes per row)
- **DATA-07**: Surface imputation recommendations in outlier results

### Strategic Improvements

- **STRAT-01**: Support ordinal and numeric variables (not just categorical via one-hot)
- **STRAT-02**: Implement traditional baseline comparisons (longstring, Mahalanobis distance, IRV)
- **STRAT-03**: Add synthetic contamination experiments for evaluation
- **STRAT-04**: Calibrate outlier scores (percentile ranks, suggested thresholds, distribution histogram)
- **STRAT-05**: Implement iterative training (remove outliers, retrain on cleaned data)
- **STRAT-06**: Add sensitivity analysis mode (show how results change without flagged rows)
- **STRAT-07**: Benchmark AE vs VAE vs PCA vs Chow-Liu tree systematically

### Core Pipeline Stabilization

- **CORE-01**: Unify data-loading return format (consistent DataFrame, dict structure)
- **CORE-02**: Extract shared data-cleaning logic into reusable function
- **CORE-03**: Consolidate dataset config definitions (YAML or Python registry)
- **CORE-04**: Fix broken test_loader.py tests (update to current DataLoader API)
- **CORE-05**: Fix CLI commands that skip data cleaning (search_hyperparameters, evaluate, etc.)
- **CORE-06**: Fix dataset config inconsistencies across duplicated blocks
- **CORE-07**: Fix COLUMNS_OF_INTEREST integer-vs-name mismatch
- **CORE-08**: Fix generate command KeyError (prior_means, prior_log_vars)
- **CORE-09**: Make Rule of 9 threshold configurable (CLI parameter)

### Model Quality

- **MODEL-01**: Add per-column outlier contribution scores computation
- **MODEL-02**: Fix numerical stability (division by log(1), Concatenate with single attribute)
- **MODEL-03**: Remove global eager mode (or guard behind DEBUG env var)
- **MODEL-04**: Fix VAE serialization issues (kl_loss_weight, warmup schedule persistence)
- **MODEL-05**: Fix data leakage in feature transformation (MinMaxScaler, OneHotEncoder fit on full data)
- **MODEL-06**: Increase default latent_space_dim to 8-16 for production

### Advanced Features

- **ADV-01**: Real-time queue position tracking ("Job #3 of 7 in queue")
- **ADV-02**: Email notifications for job completion
- **ADV-03**: Job history and re-run capability
- **ADV-04**: Webhook support for programmatic integration
- **ADV-05**: Batch processing (multiple CSVs as single job)
- **ADV-06**: Automatic schema detection beyond Rule of 9
- **ADV-07**: OAuth authentication (Google, GitHub login)
- **ADV-08**: Chow-Liu tree integration (alternative baseline method)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Core pipeline refactoring (Section 1) | Important but doesn't block web UI production use. Users primarily use web interface, not CLI. Defer to v1.1. |
| Model quality improvements (numerical stability, VAE fixes) | Valuable but current model works adequately. Not deployment blockers. Defer to v1.2. |
| Missing data masking (Section 8) | Significant research/architecture change. Current fillna("missing") strategy works for v1. Defer to v2.0. |
| Strategic concerns (Rule of 9 limitations, baseline comparisons) | Long-term research considerations, not v1 blockers. Ongoing investigation. |
| Chow-Liu tree integration | Alternative method exists in code but not integrated. Future enhancement. Defer to v2.0. |
| Containerization (docker-compose) | Nice-to-have, not required for initial production deployment. Defer to v1.1. |
| Mobile native apps | Web-first approach. No validated need for native mobile. |
| Synchronous/streaming results | 10-15 min Vertex AI jobs require async pattern. Synchronous not feasible. |
| In-browser ML training | TensorFlow.js too slow for production autoencoder training. |
| Multi-user collaboration | Adds complexity without validated need. Single-user sufficient for v1. |
| Rolling custom authentication | Use established libraries (express-session, passport) instead of custom. Security risk. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SEC-01 | Phase 1 | Complete |
| SEC-02 | Phase 1 | Complete |
| SEC-03 | Phase 1 | Complete |
| SEC-04 | Phase 1 | Complete |
| SEC-05 | Phase 1 | Complete |
| SEC-06 | Phase 1 | Complete |
| SEC-07 | Phase 1 | Complete |
| SEC-08 | Phase 1 | Complete |
| SEC-09 | Phase 1 | Complete |
| SEC-10 | Phase 1 | Complete |
| SEC-11 | Phase 1 | Complete |
| SEC-12 | Phase 1 | Complete (01-01) |
| SEC-13 | Phase 1 | Complete (01-01) |
| SEC-14 | Phase 1 | Complete |
| SEC-15 | Phase 1 | Complete |
| SEC-16 | Phase 1 | Complete (01-01) |
| WORK-01 | Phase 2 | Complete |
| WORK-02 | Phase 2 | Complete |
| WORK-03 | Phase 2 | Complete |
| WORK-04 | Phase 2 | Complete |
| WORK-05 | Phase 2 | Complete |
| WORK-06 | Phase 2 | Complete |
| WORK-07 | Phase 2 | Complete |
| WORK-08 | Phase 2 | Complete |
| WORK-09 | Phase 2 | Complete |
| WORK-10 | Phase 2 | Complete |
| WORK-11 | Phase 2 | Complete |
| WORK-12 | Phase 2 | Complete |
| WORK-13 | Phase 2 | Complete |
| WORK-14 | Phase 2 | Complete |
| FE-01 | Phase 3 | Complete |
| FE-02 | Phase 3 | Complete |
| FE-03 | Phase 3 | Complete |
| FE-04 | Phase 3 | Complete |
| FE-05 | Phase 3 | Complete |
| FE-06 | Phase 3 | Complete |
| FE-07 | Phase 3 | Complete |
| FE-08 | Phase 3 | Complete |
| FE-09 | Phase 3 | Complete |
| FE-10 | Phase 3 | Complete |
| FE-11 | Phase 3 | Complete |
| FE-12 | Phase 3 | Complete |
| FE-13 | Phase 3 | Complete |
| FE-14 | Phase 3 | Complete |
| FE-15 | Phase 3 | Complete |
| FE-16 | Phase 3 | Complete |
| FE-17 | Phase 3 | Complete |
| FE-18 | Phase 3 | Complete |
| FE-19 | Phase 3 | Complete |
| FE-20 | Phase 3 | Complete |
| FE-21 | Phase 3 | Complete |
| FE-22 | Phase 3 | Complete |
| OPS-01 | Phase 4 | Complete |
| OPS-02 | Phase 4 | Complete |
| OPS-03 | Phase 4 | Complete |
| OPS-04 | Phase 4 | Complete |
| OPS-05 | Phase 4 | Complete |
| OPS-06 | Phase 4 | Complete |
| OPS-07 | Phase 4 | Complete |
| OPS-08 | Phase 4 | Complete |
| OPS-09 | Phase 4 | Complete |
| OPS-10 | Phase 4 | Complete |
| OPS-11 | Phase 4 | Pending |
| OPS-12 | Phase 4 | Pending |
| OPS-13 | Phase 4 | Complete |
| OPS-14 | Phase 4 | Complete |
| GH-01 | Phase 4 | Complete |
| GH-02 | Phase 4 | Complete |
| GH-03 | Phase 4 | Complete |
| GH-04 | Phase 4 | Complete |
| GH-05 | Phase 4 | Complete |

**Coverage:**
- v1.0 requirements: 71 total
- Mapped to phases: 71 (100%)
- Unmapped: 0

---
*Requirements defined: 2026-03-24*
*Last updated: 2026-03-24 after roadmap creation*
