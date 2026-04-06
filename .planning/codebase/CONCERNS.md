# Codebase Concerns

**Analysis Date:** 2026-01-23

## Tech Debt

**Large Monolithic Files:**
- Issue: `main.py` is 1,661 lines with 9 CLI commands containing duplicated logic
- Files: `main.py`
- Why: Rapid development without refactoring
- Impact: Hard to maintain, test, and understand
- Fix approach: Extract common dataset configuration logic into utilities

**Commented-Out Code:**
- Issue: 100+ lines of commented-out alternative implementations
- Files: `model/loss.py` (lines 20-38, 69-79, 101-141), `worker.py` (lines 106-213)
- Why: Kept for reference during experimentation
- Impact: Makes code harder to read, confuses intent
- Fix approach: Remove commented code, use git history for reference

**Inconsistent Logging:**
- Issue: Mix of `print()` statements and `logging` calls
- Files: `main.py` (13+ debug prints), `worker.py`, `utils.py`
- Why: Quick debugging during development
- Impact: Inconsistent log output, hard to control verbosity
- Fix approach: Replace all print() with logger calls, configure log levels

**Wrong Logger Initialization:**
- Issue: Using `logging.Logger(__name__)` instead of `logging.getLogger(__name__)`
- Files: `main.py` (line 53-55), `train/trainer.py`
- Why: Misunderstanding of logging API
- Impact: Loggers without proper handlers, logs may not appear
- Fix approach: Use `logging.getLogger(__name__)` consistently

## Known Bugs

**Bare Except Clauses:**
- Symptoms: All exceptions silently caught, including KeyboardInterrupt
- Trigger: Any exception in protected code blocks
- Files: `main.py` (line 247, 578), `worker.py` (line 578)
- Workaround: None - exceptions are swallowed
- Root cause: Using `except:` instead of `except Exception as e:`
- Fix: Add specific exception types or at minimum `except Exception`

**Incomplete Worker Implementation:**
- Symptoms: Worker only dispatches to Vertex AI, doesn't process locally
- Trigger: Running worker.py
- File: `worker.py` (lines 106-213 commented out)
- Workaround: Use Vertex AI for all processing
- Root cause: Processing logic was commented out
- Fix: Either remove dead code or implement properly

## Security Considerations

**Exposed Service Account Key:**
- Risk: GCP credentials committed to repository
- File: `frontend/service-account-key.json`
- Current mitigation: None
- Recommendations:
  - Immediately rotate key in GCP console
  - Add to `.gitignore`
  - Use environment variables or secret manager

**Environment File in Git:**
- Risk: `.env` file may contain secrets
- File: `.env`
- Current mitigation: `.env.example` exists as template
- Recommendations: Verify `.env` is in `.gitignore`, review commit history

**TypeScript Strict Mode Disabled:**
- Risk: Type errors not caught at compile time
- File: `frontend/tsconfig.json` (`strict: false`)
- Current mitigation: None
- Recommendations: Enable strict mode, fix type errors

## Performance Bottlenecks

**Inefficient DataFrame Iteration:**
- Problem: Using `.iterrows()` in evaluation loop
- File: `utils.py` (lines 153-178, `evaluate_errors` function)
- Measurement: Slow on large DataFrames (O(n) Python loop)
- Cause: `.iterrows()` creates Python objects per row
- Improvement path: Use vectorized operations (`.apply()`, boolean indexing)

**Repeated Column Lookups:**
- Problem: Column existence checks repeated without caching
- File: `main.py` (lines 228-232, similar in other functions)
- Measurement: Minor but repeated O(n) lookups
- Cause: No caching of column lists
- Improvement path: Cache column lookups once per dataset

## Fragile Areas

**Data Loader Return Format:**
- File: `main.py` (lines 173-193)
- Why fragile: Returns vary between DataFrame, (DataFrame, dict), or ((DataFrame, metadata), dict)
- Common failures: TypeError when unpacking wrong format
- Safe modification: Check return type explicitly before unpacking
- Test coverage: No tests for different return formats

**Config File Access:**
- File: `main.py` (lines 260-261, 350-351)
- Why fragile: No try/except around file opening
- Common failures: FileNotFoundError crashes program
- Safe modification: Add file existence check
- Test coverage: No tests for missing config scenarios

## Scaling Limits

**Single Worker:**
- Current capacity: One Pub/Sub subscription listener
- Limit: Sequential job processing
- Symptoms at limit: Job queue backlog
- Scaling path: Multiple worker instances with Pub/Sub automatic load balancing

## Dependencies at Risk

**TensorFlow/Keras Version:**
- Risk: TensorFlow 2.15.1 + separate Keras 2.15.0 can conflict
- File: `requirements.txt`
- Impact: Import errors, version mismatches
- Migration plan: Use `tensorflow.keras` consistently, remove separate keras package

**Pinned Dependencies:**
- Risk: No upper bounds, builds may break with future versions
- File: `requirements.txt` (123 packages with exact versions)
- Impact: Potential security vulnerabilities in old versions
- Migration plan: Use version ranges, run security audits

## Missing Critical Features

**No Authentication:**
- Problem: API endpoints have no authentication
- Current workaround: Deploy on private network
- Blocks: Public deployment
- Implementation complexity: Medium (add JWT or API key auth)

**No Job Cancellation:**
- Problem: Cannot cancel running Vertex AI jobs from UI
- Current workaround: Cancel manually in GCP console
- Blocks: User control over long-running jobs
- Implementation complexity: Low (add cancel endpoint)

## Test Coverage Gaps

**CLI Commands:**
- What's not tested: All `main.py` CLI commands (train, evaluate, find_outliers, etc.)
- Files: `main.py`
- Risk: Regression in primary user interface
- Priority: High
- Difficulty: Medium - need to mock file I/O and ML training

**Worker Processing:**
- What's not tested: `worker.py` Pub/Sub handling and Vertex AI dispatch
- File: `worker.py`
- Risk: Job processing failures go unnoticed
- Priority: High
- Difficulty: High - need to mock GCP services

**Utils Functions:**
- What's not tested: `utils.py` helper functions
- File: `utils.py`
- Risk: Utility bugs affect multiple callers
- Priority: Medium
- Difficulty: Low - pure functions easy to test

**End-to-End Flow:**
- What's not tested: Full upload → train → results pipeline
- Risk: Integration bugs between components
- Priority: High
- Difficulty: High - requires test infrastructure

## Documentation Gaps

**Architecture Documentation:**
- What's missing: System design, data flow diagrams
- Files affected: All
- Impact: Difficult onboarding, unclear component responsibilities

**API Documentation:**
- What's missing: OpenAPI/Swagger spec for Express endpoints
- File: `frontend/server/`
- Impact: Frontend developers must read code to understand API

**Configuration Reference:**
- What's missing: Documentation of YAML config keys
- Files: `config/*.yaml`
- Impact: Trial-and-error to understand valid configurations

---

*Concerns audit: 2026-01-23*
*Update as issues are fixed or new ones discovered*
