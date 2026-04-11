# TASKS.md -- Product Launch for Survey Outlier Detection

## 1. Stabilize the Core Pipeline

### ~~1.1 Unify data-loading return format~~ DONE
All `DataLoader` methods now consistently return `(DataFrame, metadata_dict)` where metadata always has `"variable_types"` and `"ignored_columns"` keys. Fixed `load_eval_dataset()` (the only outlier, which returned a nested `((df, base_df), types)` tuple) to use `prepare_original_dataset()` like all other loaders. Removed the isinstance workaround code from both `train()` and `find_outliers()` in `main.py`.

### ~~1.2 Extract shared data-cleaning logic~~ DONE
Extracted `prepare_for_model(project_data, variable_types)` in `main.py` that performs the full pipeline: fillna("missing") → astype(str) → Rule-of-9 filter → sync variable_types → Table2Vector vectorization → float32 conversion → cardinality computation. Returns `(cleaned_df, vectorized_df, vectorizer, cardinalities)`. Now called from `train()`, `find_outliers()`, and `run_training_pipeline()` — eliminated ~80 lines of duplicated cleaning code.

### ~~1.3 Consolidate dataset config definitions~~ DONE
Replaced the inline dataset config blocks in `evaluate_on_condition` (~340 lines) and `pca_baseline` (~470 lines) with calls to `define_necessary_elements()` from `utils.py` — the single source of truth for all dataset configs. All CLI commands (`train`, `find_outliers`, `evaluate`, `evaluate_on_condition`, `pca_baseline`) now use the same function. Note: the configs that were in `evaluate_on_condition`/`pca_baseline` had slight differences from `utils.py` (documented in task 1.7); those discrepancies should be resolved in task 1.7 by verifying which values are correct.

### ~~1.4 Fix broken test_loader.py tests~~ DONE
The `TestDataLoaderAPI` tests were already rewritten in the prior PR to use the current `DataLoader` API with synthetic data. The remaining issue was the `TestDataLoaderSADCRegression` test which unpacked `load_2017()` as `(df, var_types)` instead of `(df, metadata)` — fixed to use `metadata["variable_types"]` and added assertions for both `variable_types` and `ignored_columns` keys.

### ~~1.5 Fix broken test_loss.py tests~~ DONE
Fixed in a prior PR: removed the nonexistent `CustomCategoricalCrossentropyVAE` import, rewrote tests to only use `CustomCategoricalCrossentropyAE`, and added `test_get_config_includes_percentile` to verify the serialization fix from task 1.11.

### ~~1.6 Fix CLI commands that skip data cleaning~~ DONE
Fixed all CLI commands to use the same `prepare_for_model()` cleaning pipeline as `train`/`find_outliers`:
- **`search_hyperparameters`**: Already fixed in prior refactoring (uses `prepare_for_model`)
- **`evaluate`**: Replaced manual `Table2Vector` + `vectorize_table` with `prepare_for_model()` call, ensuring the same fillna/astype(str)/Rule-of-9 pipeline as training
- **`pca_baseline`**: Same fix — uses `prepare_for_model()` after dropping conditioning columns. Also added None guard for `--column_to_condition` and `--outlier_value` to prevent `None.split(",")` crash
- **`evaluate_on_condition`**: Added same None guard for required options. This command doesn't vectorize data (it reads pre-computed errors.csv), so no cleaning pipeline change needed
- Also fixed bare `except:` clause in `find_outliers` to `except Exception:`

### ~~1.7 Fix dataset config inconsistencies across duplicated blocks~~ DONE
The column configs in `evaluate_on_condition` / `pca_baseline` previously differed from `define_necessary_elements` (utils.py) because the evaluation blocks included extra attention-check / screening columns as gold labels. These columns are intentionally excluded from COLUMNS_OF_INTEREST in `define_necessary_elements` (they would leak ground truth into training). The fix: modified `find_outlier_data()` and `find_outlier_data_sadc_2017()` in `dataset/loader.py` to temporarily disable COLUMNS_OF_INTEREST filtering when loading data for evaluation, so attention-check columns are always accessible regardless of the training config.

### ~~1.8 Fix `COLUMNS_OF_INTEREST` integer-vs-name mismatch~~ DONE
Fixed `load_original_data()` to handle both integer positional indices and string column names in COLUMNS_OF_INTEREST. Integer indices now use positional lookup (`original_df.columns[valid]`) while string names use membership check. Previously, integer indices were compared against string column names and never matched, making the filter a no-op.

### ~~1.9 Fix `run_training_pipeline` typo~~ DONE
Fixed typo: `"guassian"` → `"gaussian"` in the default parameter of `run_training_pipeline()`.

### ~~1.10 Fix `generate` command~~ DONE
Fixed `KeyError` on `prior_means`/`prior_log_vars` — the VAE's `get_config()` never included these keys. Now computes latent-space statistics by running the encoder on the training data (`tf.reduce_mean` of `z_mean`/`z_log_var` across all rows).

### ~~1.11 Fix `CustomCategoricalCrossentropyAE.get_config` missing `percentile`~~ DONE
Fixed in a prior PR: `get_config()` now includes `percentile` in its return dict, ensuring the value is preserved across model save/load cycles. Test coverage added in `test_get_config_includes_percentile`.

### ~~1.12 Remove DEBUG print statements~~ DONE
All `print("DEBUG: ...")` statements in `main.py` have been replaced with proper `logger.debug()`, `logger.warning()`, and `logger.error()` calls. Emoji-prefixed error prints were also cleaned up.

## 2. Harden the Upload Pipeline (Web UI)

### ~~2.1 Implement end-to-end upload processing in worker.py~~ DONE
`worker.py` now has two modes: `--mode=local` (default) processes uploads entirely locally (download from GCS, train autoencoder, score outliers, write to Firestore), and `--mode=vertex` dispatches to Vertex AI. The local mode enables running the demo without any Vertex AI setup. The worker also writes a `"processing"` status to Firestore before starting, so the frontend can show progress.

### ~~2.2 Handle arbitrary user CSV uploads robustly~~ DONE
Added `tests/test_upload_pipeline.py` with 13 integration tests that exercise `DataLoader.load_uploaded_csv()` on the edge cases called out in this task: mixed numeric + categorical columns, special characters in column names (spaces, hyphens, dots, slashes), unicode column names and values (including emoji), wide datasets (120 columns), mostly-missing datasets (>90% NaN), and completely-numeric datasets (all columns binned through `convert_to_categorical`). Each test asserts the result is consumable by `Table2Vector` end-to-end (no NaN in the one-hot matrix).

Fixed two pre-existing bugs in `DataLoader.prepare_original_dataset()` that the new tests exposed:
- **Rule of 9 was one-sided**: the filter only dropped columns with `n_unique > 9` but kept single-value columns, even though `CLAUDE.md` specifies "Columns with more than 9 unique values **or only 1 unique value** are dropped". Fixed to `1 < n_unique <= 9`, matching `worker.py`'s inline cleaning and `main.prepare_for_categorical`. All-NaN numeric columns (which `convert_to_categorical` turns into a constant `"NA"` column) are now correctly dropped.
- **NaN in categorical columns became the literal string `"nan"`**: the old path did `astype(str)` without filling NaN, producing unhelpful `col__nan` one-hot buckets. Fixed by filling NaN with `"missing"` after `convert_to_categorical` (which already handles NaN for numerics via its own "missing" bin), matching worker.py and `main.prepare_for_categorical` behaviour.

Also removed the `@unittest.skip` marker from `tests/test_upload_pipeline.py` since the upload path is now usable as a baseline. Metadata shape is verified by `test_metadata_is_consistent_with_clean_df` (variable_types keys match clean_df columns, ignored columns are disjoint from kept columns).

### ~~2.3 Add input validation and error reporting~~ DONE
Replaced the old "write `str(e)` to Firestore" path with structured error reporting end-to-end. The worker now writes `{error, errorCode, errorType}` to the job document, and the frontend maps each stable `errorCode` to a clean, user-facing heading + message. Highlights:

- **`worker.py`**: Added `ErrorCode` and `ErrorType` enums and `UploadValidationError` (a `ValueError` subclass carrying an `error_code`). Added `mark_job_error(job_ref, job_id, message, error_code, error_type)` that writes the structured payload through the existing state-machine transaction and tolerates invalid transitions (e.g. when a terminal state has already been reached).
- **`validate_csv`**: Every failure path now raises `UploadValidationError` with a specific `ErrorCode` (`CSV_TOO_LARGE`, `CSV_EMPTY`, `CSV_ENCODING`, `CSV_PARSE`, `CSV_INCONSISTENT_COLUMNS`, `CSV_TOO_FEW_ROWS`, `CSV_TOO_FEW_COLUMNS`). The exception still inherits from `ValueError`, so existing `pytest.raises(ValueError, match=...)` call sites in `tests/test_csv_validation.py` keep working unchanged.
- **`process_upload_local`**: Classifies each failure point into a dedicated error code: validation errors use the raised code; DataLoader failures become `LOAD_FAILURE`; the "all columns dropped by Rule of 9" branch becomes `NO_USABLE_COLUMNS` (and attaches the `stats` payload so the UI can still show which columns were dropped); training failures become `TRAINING_FAILURE`; the outer catch-all writes `INTERNAL_ERROR` with a generic user-facing message instead of a raw exception string. An unhandled `UploadValidationError` that escapes inner handlers is also routed through `mark_job_error` so the code is preserved.
- **`process_upload_vertex`**: Generic failures in the Vertex dispatch path also emit `INTERNAL_ERROR` via `mark_job_error` instead of leaking raw SDK error strings.
- **Frontend**: Added `frontend/client/utils/jobErrors.ts` with a `resolveJobError()` helper that maps every `ErrorCode` to a `{heading, fallbackMessage}` pair and falls back by `errorType` if the code is unknown. `JobProgress.tsx` now shows the heading + message + error code monospace line in its red error card; `Index.tsx`'s polling error handler uses the same helper. `useJobPolling.ts` and `utils/api.ts` surface `errorCode`/`errorType` on the `JobStatus` interface.
- **Tests**: New `tests/test_error_reporting.py` with 16 tests covering `UploadValidationError`/`ValueError` compatibility, each `validate_csv` error code, `mark_job_error` happy path, enum-to-string serialization, plain-string inputs, the "invalid transition is swallowed silently" case, and a drift-guard on the `ErrorCode`/`ErrorType` enum values. All 99 worker-level tests pass (new + existing).

### ~~2.4 Add API authentication~~ DONE
Full session-based authentication landed as part of the Phase 01 security foundation. `frontend/server/middleware/auth.ts` configures a Passport `LocalStrategy` (email + bcrypt-hashed password) backed by a Firestore `users` collection, with `requireAuth` middleware that returns 401 on unauthenticated requests. Sessions are signed with a `SESSION_SECRET` that envalid rejects at startup if shorter than 32 chars (`frontend/server/config/env.ts`) and stored in Firestore via `firestore-store` (`frontend/server/config/session.ts`). `requireAuth` is now applied to every mutating / data-bearing endpoint: `/api/upload`, `/api/jobs/upload-url`, `/api/jobs/start-job`, `/api/jobs/job-status/:id`, `/api/jobs/:id/export`, `DELETE /api/jobs/:id`, and `DELETE /api/jobs/:id/files`. Auth routes themselves (`/api/auth/signup`, `/login`, `/logout`, `/me`, `/request-reset`, `/reset-password`) are protected by an independent `authLimiter` rate limit. Test coverage in `frontend/server/__tests__/middleware/auth.test.ts`, `__tests__/routes/auth.test.ts`, and `__tests__/integration/security.test.ts`.

### ~~2.5 Fix security issues in the upload/API pipeline~~ DONE
All six bullets addressed by the Phase 01 security foundation plus this PR's multer fileFilter addition:
- **Open CORS** → fixed in `frontend/server/middleware/security.ts`: `corsConfig` now uses an explicit allowlist seeded from `FRONTEND_URL` (normalized via `new URL(...).origin` so trailing slashes don't break matching), and `csrfOriginCheck` rejects mutating cross-origin requests from unlisted origins — needed because session cookies use `sameSite=none`, so CORS alone can't stop CSRF.
- **Path traversal via filename** → fixed by `generateSafeFilename(userId)` in `frontend/server/utils/fileValidation.ts`: the user-provided `originalname` is discarded entirely and replaced with `uploads/{userId}/{uuid}.csv`. `sanitizePath()` provides a defense-in-depth helper for any future path-joining code. Additionally, `/api/jobs/start-job` rejects `gcsFileName` values that don't start with the caller's `uploads/{userId}/` prefix (routes/jobs.ts).
- **No file-type validation on upload** → fixed in two layers: (1) the new `csvUploadFileFilter` exported from `frontend/server/index.ts` rejects non-`.csv` extensions and unsupported MIME types *before* multer reads the file into memory (returns 400, not 500), and (2) `validateCSVContent` in `utils/fileValidation.ts` still runs inside the route using `file-type` magic-byte detection plus CSV structural checks for the authoritative validation. New unit tests in `frontend/server/__tests__/utils/uploadFileFilter.test.ts` cover the filter's accept/reject paths (9 cases).
- **No rate limiting** → fixed in `frontend/server/middleware/rateLimits.ts`: `uploadLimiter` (5/15min) on `/api/upload` and `/start-job`, `uploadUrlLimiter` (independent 5/15min bucket) on `/upload-url` so a 3-step upload doesn't halve the budget, `pollLimiter` (60/min) on `/job-status`, `downloadLimiter` (10/hr) on `/export`, `authLimiter` (10/15min by IP) on auth routes. All keyed by user-id-or-IP with IPv6-safe key generation.
- **Signed URL endpoint unauthenticated** → fixed: `/api/jobs/upload-url` now requires `requireAuth` + `uploadUrlLimiter` + `validateUploadUrl`. The legacy `routes/demo.ts` file no longer exists.
- **Job IDs passed to Firestore without validation** → fixed in `frontend/server/middleware/validation.ts`: `validateJobId`, `validateUploadUrl`, `validateStartJob`, `validateSignup`, `validateLogin`, `validateRequestReset`, `validateResetPassword` all use express-validator to reject non-UUID job IDs and malformed request bodies before they reach Firestore. Additionally, every read/mutate path in `routes/jobs.ts` now verifies `job.userId === req.user.id` and returns 404 otherwise, so IDOR attacks return "not found" rather than leaking existence.

### ~~2.6 Fix worker reliability issues~~ DONE
All three worker reliability concerns fixed:
- **Ack race condition** → fixed by the new `AckExtender` class in `worker.py`. It uses `threading.Timer` to call `message.modify_ack_deadline(interval + 10)` every 60 seconds during long jobs, preventing the Pub/Sub ack deadline from expiring during multi-minute training runs. The same heartbeat also refreshes a `claimedAt` timestamp on the job doc so a concurrent duplicate delivery can distinguish "another worker is alive and working" from "original worker crashed — safe to take over". Both `process_upload_local` and `process_upload_vertex` instantiate an `AckExtender(message, job_ref=job_ref)` before any heavy work and stop it in `finally`. Tests: `tests/test_ack_extension.py` (7 cases).
- **No message field validation** → fixed by the `PubSubMessage` Pydantic model and `validate_message()` helper in `worker.py`. `jobId`, `bucket`, and `file` are all required non-empty strings. The `callback()` ack-drops `ValueError` (poison messages) so they don't loop through redelivery indefinitely, and non-mapping JSON inputs (lists, nulls, bare strings/numbers) are normalized to `ValueError` so they take the same ack-drop path instead of crashing with `TypeError` and triggering nacks. Tests: `tests/test_message_validation.py` (12 cases).
- **No env var validation at startup** → fixed by `validate_environment()` called from the `__main__` block before `subscriber.subscribe()`. Reads `GOOGLE_CLOUD_PROJECT`, `GCS_BUCKET_NAME`, and `PUBSUB_SUBSCRIPTION_ID` fresh via `os.getenv()` (for testability) and calls `sys.exit(1)` with a clear error if any are missing. Tests: `tests/test_env_validation.py` (4 cases).

### ~~2.7 Remove hardcoded GCP identifiers from source code~~ DONE
All three hardcoded GCP identifiers now read from environment variables with the original values as defaults:
- `worker.py`: Staging bucket reads from `VERTEX_STAGING_BUCKET` env var (default: `gs://autoencoders-census-staging`)
- `worker.py`: Service account reads from `VERTEX_SERVICE_ACCOUNT` env var (default: `203111407489-compute@developer.gserviceaccount.com`)
- `train/task.py`: Project ID reads from `GOOGLE_CLOUD_PROJECT` env var (default: `autoencoders-census`)
Also added `import os` to `train/task.py` and documented the new optional env vars in `worker.py`'s docstring.

### ~~2.8 Fix inconsistent outlier scoring between CLI and web~~ DONE
Extracted a shared `compute_reconstruction_error(data, predictions, attr_cardinalities)` helper in `evaluate/outliers.py` that wraps `VAE.reconstruction_loss` (per-attribute categorical crossentropy normalized by `log(K)`) and returns a numpy array. This is now the single source of truth for outlier scoring, used by all three code paths:
- **CLI (`find_outliers`)**: `get_outliers_list` was already calling `VAE.reconstruction_loss` directly; refactored to go through the new helper so the CLI and the other two paths cannot drift apart.
- **Local worker (`worker.py:process_upload_local`)**: replaced `np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)` with `compute_reconstruction_error(vectorized_df, reconstruction, cardinalities)`.
- **Vertex AI container (`train/task.py`)**: same replacement.

Added `tests/test_reconstruction_error.py` with 9 tests: unit tests for the helper (perfect-reconstruction → zero error, worse predictions → higher error, DataFrame/ndarray interchangeability, numpy return type, direct equivalence to `VAE.reconstruction_loss`, strictly monotonic error for rows with 0/1/2 wrong attributes), a structural regression guard that fails if `worker.py` or `train/task.py` reintroduces the legacy MSE (via source inspection), a sanity check that the new score is genuinely different from the old MSE (correlation < 0.9999 on synthetic data), and an end-to-end check that `get_outliers_list` populates its `error` column from the shared helper.

## 3. Improve Model Quality and Configurability

### 3.1 Make the "Rule of 9" threshold configurable
The max-unique-values threshold of 9 is hardcoded in `DataLoader.prepare_original_dataset()` (line 439) and duplicated in `main.py` `train`/`find_outliers` commands. Make this a CLI parameter and config option so users can adjust it for their specific datasets.

### 3.2 Support custom model configs for uploaded data
When a user uploads a CSV through the web UI, the system should either auto-select reasonable hyperparameters based on the data shape or allow the user to choose a config preset. Currently, the uploaded data path does not specify which config to use.

### 3.3 Add per-column outlier contribution scores
The `get_outliers_list()` function returns an aggregate reconstruction error per row. For interpretability, also compute and return per-column reconstruction error so users can see *which* survey questions a flagged respondent answered anomalously.

### ~~3.4 Benchmark model variants~~ DONE (initial results)
Initial AE vs Chow-Liu comparison completed on SADC 2015 and 2017 — see task 10.5 for full results. Summary: comparable ROC AUC (~0.71-0.76), AE better at top-of-list precision on 2017, Chow-Liu more robust on 2015 and dramatically faster. Remaining work: extend to non-SADC datasets (Pennycook, bot_bot_mturk, etc.) once those datasets are available locally, and benchmark the VAE variant.

### ~~3.5 Fix numerical stability issues in loss computation~~ DONE
- **Division by log(1) = 0**: `model/loss.py` and `model/base.py` already guarded with `max(int(categories), 2)`. Fixed the remaining unguarded location in `AutoencoderModel.__init__` (`model/autoencoder.py`) which computed `np.log(cardinality)` without clamping — now uses `np.log(max(cardinality, 2))`.
- **`Concatenate()` with single attribute**: Fixed all three locations (`model/autoencoder.py` `build_decoder` and `build_decoder_hp`, `model/layers.py` `build_decoder`) to skip `Concatenate` and pass through the single tensor directly when there is only one attribute.
- Added tests: `test_single_category_attribute_no_division_by_zero` (loss), `test_log_cardinalities_guards_cardinality_one` and `test_single_attribute_decoder_builds` (autoencoder).

### ~~3.6 Remove global eager mode~~ DONE
Removed `tf.config.run_functions_eagerly(True)` from module level in `model/base.py` and `model/variational_autoencoder.py`. This was a debug leftover that forced all TensorFlow functions to run eagerly, causing 2-10x training slowdown. Graph mode is now used by default, which is the standard TensorFlow behavior.

### 3.7 Fix VAE serialization issues (LOW PRIORITY)
- `kl_loss_weight` is overwritten to `0.0` in the constructor (model/base.py:36-40) for warmup scheduling. But `get_config()` (model/base.py:269-270) calls `.numpy()` on `self.kl_loss_weight`, which is a Python float (not a tensor) before the first training step, raising `AttributeError`.
- After deserialization via `from_config`, the warmup schedule restarts from 0 regardless of where training left off.
- VAE is not actively used; fix only if VAE work resumes.

### ~~3.8 Fix data leakage in feature transformation~~ DONE
Added a proper `fit()` / `transform()` API to `Table2Vector` so that encoders and scalers are fitted on training data only. The legacy `vectorize_table()` still works for scoring/evaluation where no train/test split is needed. All training pipelines (`Trainer.train`, `run_training_pipeline`, `worker.py`, `train/task.py`) now split *before* vectorizing: clean → split → fit on train → transform both. Three bugs fixed:
- **MinMaxScaler leakage**: `fit()` learns scaler ranges on training data only; `transform()` applies them without refitting.
- **OneHotEncoder leakage**: Same — encoder categories come from training split only. Unseen test-set categories get all-zeros via `handle_unknown='ignore'`.
- **Index misalignment**: One-hot encoded DataFrames now explicitly receive the source DataFrame's index (`index=vectorized_df.index`), preventing silent NaN introduction from `pd.concat` with mismatched indices. Same fix applied in `tabularize_vector()` and `evaluate/outliers.py`.
Added 15 new tests in `tests/features/test_data_leakage.py` covering index preservation, fit/transform API, no-leakage guarantees, and the `prepare_for_training()` integration.

## 4. Frontend Polish

### 4.1 Display per-column outlier explanations
When the results table shows flagged outliers, include a breakdown of which columns contributed most to the high reconstruction error (depends on task 3.3).

### 4.2 Add progress indicator for long-running jobs
Vertex AI jobs take 10-15 minutes. The frontend polls Firestore for completion. Add a progress state (e.g., "Preprocessing", "Training", "Scoring") that the worker writes to Firestore so users see meaningful status updates instead of just "Processing...".

### 4.3 Allow downloading results as CSV
Add an export button on the results page that lets users download the outlier-scored data as a CSV file.

### 4.4 Support job cancellation
Users currently cannot cancel running jobs from the UI (noted in README). Add a cancellation endpoint that marks the job as cancelled in Firestore and, if using Vertex AI, cancels the cloud job.

### 4.5 Fix frontend build blockers
The frontend cannot currently compile or deploy:
- **Missing `lib/utils.ts`**: 48 components import `cn` from `@/lib/utils` (the standard shadcn/ui utility combining `clsx` + `tailwind-merge`), but the file does not exist.
- **Missing `react-router-dom` dependency**: `NotFound.tsx` imports from it, but it is not in `package.json`.
- **Missing `serverless-http` dependency**: `netlify/functions/api.ts` imports it, but it is not in `package.json`.
- **Missing `build:client` script**: CLAUDE.md and the Netlify deployment guide (7.8) reference `npm run build:client`, but `package.json` only has `build`.
- **Missing `dev:server` script**: CLAUDE.md references `npm run dev:server` to start the Express server, but it is not in `package.json`.

### 4.6 Fix frontend runtime issues
- **No React error boundary**: Any render error white-screens the entire app with no recovery. Add an error boundary around the root component.
- **CSV parser reads entire file into memory** (csv-parser.ts:77): For preview that only needs 20 rows, this is wasteful and can crash the browser tab on large files.
- **Polling `useEffect` depends on `toast`** (Index.tsx:26-48): `toast` is a new function reference on every render, causing the polling interval to tear down and restart constantly.
- **Click-upload skips file type validation** (Dropzone.tsx:55-59): Drag-and-drop validates `.csv` but the file input change handler does not.

### 4.7 Fix frontend operational issues
- **TypeScript strict mode disabled** (tsconfig.json): `strict`, `strictNullChecks`, `noImplicitAny` are all `false`. This undermines the value of using TypeScript.
- **Duplicate GCP client instances**: `index.ts`, `routes/jobs.ts`, and `routes/demo.ts` each create separate Storage/Firestore/PubSub clients. Create once and share.
- **Two overlapping job-status routes**: `index.ts:122` and `routes/jobs.ts:79` both handle `/api/jobs/job-status/:id` with different 404 behavior.
- **Port mismatch**: `server/start.ts` uses port 3000, but CLAUDE.md says 5001.

## 5. Testing and CI

### ~~5.1 Get all existing tests passing~~ DONE
Fixed all existing test failures:
- **`test_loader.py`**: Rewrote with current `DataLoader` API using synthetic CSV data instead of stale `DATASET_URL_2015`/`DATASET_URL_2017` class attributes. Tests now cover `prepare_original_dataset()`, Rule-of-9 filtering, and NaN handling.
- **`test_loss.py`**: Removed nonexistent `CustomCategoricalCrossentropyVAE` import and dead tests. Added `test_get_config_includes_percentile` to verify the `get_config()` fix from task 1.11.
- Also fixed `model/loss.py` `get_config()` to include the `percentile` parameter (task 1.11), which was discovered while fixing tests.

### ~~5.2 Add integration tests for the CLI pipeline~~ DONE
Added `tests/test_integration_pipeline.py` with end-to-end tests on synthetic data (no GCP dependency):
- `test_full_training_pipeline`: synthetic CSV → `DataLoader` → `Table2Vector` → autoencoder train → reconstruction → outlier scoring. Verifies output shape, error ordering, and that injected random rows score higher than clean rows.
- `test_vectorization_roundtrip`: verifies one-hot encoding produces correct dimensionality.
- `test_outlier_scoring_ranks_random_rows_higher`: verifies that deliberately noisy rows receive higher reconstruction error than consistent rows.

### 5.3 Add tests for the web upload path
Placeholder tests exist in `tests/test_upload_pipeline.py` but are blocked by known bugs in the upload path (tasks 2.2-2.3). Write full tests for `DataLoader.load_uploaded_csv()` with various CSV payloads (valid, empty, malformed, unicode). Mock GCS and Firestore interactions to test the worker pipeline end-to-end.

### ~~5.4 Set up CI pipeline~~ DONE
Added `.github/workflows/ci.yml` — a GitHub Actions workflow that:
- Triggers on push and pull requests
- Runs on `ubuntu-latest` with Python 3.10
- Installs dependencies from `requirements.txt` (includes `pytest`)
- Runs `python -m pytest tests/ -v`

## 6. Documentation and Deployment

### 6.1 Write user-facing documentation
Create a short guide explaining: what kind of data to upload, what the outlier scores mean, how to interpret per-column contributions, and what the "Rule of 9" filter does to their data.

### 6.2 Add a sample dataset and walkthrough
Include a small anonymized sample CSV in the repo (or a script to generate synthetic survey data) along with a step-by-step tutorial running the CLI pipeline and interpreting results.

### 6.3 Containerize the full stack
Create a `docker-compose.yml` that runs the React frontend, Express API, and Python worker together. This enables one-command local deployment without needing three terminals.

### ~~6.4 Document cloud deployment~~ DONE
See section 7 below.

### 6.5 Fix Dockerfile and add .dockerignore
- **Dockerfile installs unpinned versions** (Dockerfile:11-19): Packages are installed without version pins, diverging from `requirements.txt`. Should use `COPY requirements.txt . && pip install -r requirements.txt` instead.
- **No `.dockerignore`**: `COPY . .` puts `.env`, `.git/`, `node_modules/`, `cache/` into the image — a security risk (leaking secrets) and a performance issue (bloated image).

### 6.6 Fix operational hygiene issues
- **`logging.Logger()` instead of `logging.getLogger()`** (main.py:53, train/trainer.py:9): Creates detached loggers outside the standard hierarchy, ignoring `basicConfig()` and root logger configuration.
- **Bare `except:` clause** (main.py:570): Catches everything including `KeyboardInterrupt` and `SystemExit`. Should be `except Exception:`.
- **Unused `google.cloud.storage` import** (main.py:32): Forces GCP dependency for all CLI commands, even purely local operations. Make it a lazy import.
- **`ranx` imported at top level** (utils.py:9): Heavy dependency (~15 transitive packages) only used in one evaluation function. Make it a lazy import.
- **`plt.show()` blocks in headless environments** (utils.py:199, 217, 293): Should use `plt.savefig()` or be made configurable. Matplotlib figures are also never closed, causing memory leaks.
- **`.gitignore` blanket-ignores `*.json`** (.gitignore:67-70): Too broad — catches `components.json`, `package-lock.json`, and any future JSON configs. Intent is to prevent committing service account keys; use a more targeted pattern.
- **`.env` explicitly un-ignored** (frontend/.gitignore:27): `!.env` rule means the `.env` file with `GOOGLE_APPLICATION_CREDENTIALS` path is committed to git.

## 7. Cloud Deployment Guide (GCP project: `autoencoders-census`)

Full setup instructions for the demo website on GCP.

### 7.1 GCP APIs to enable

```bash
gcloud services enable \
  storage.googleapis.com \
  firestore.googleapis.com \
  pubsub.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  --project=autoencoders-census
```

### 7.2 Google Cloud Storage

Create the two required buckets:

```bash
# Bucket for uploaded CSV files (referenced in frontend/.env as GCS_BUCKET_NAME)
gsutil mb -p autoencoders-census -l us-central1 gs://autoencoder_data

# Staging bucket for Vertex AI (referenced in worker.py)
gsutil mb -p autoencoders-census -l us-central1 gs://autoencoders-census-staging
```

Set CORS on the upload bucket so the browser can upload directly via signed URLs:

```bash
cat > /tmp/cors.json << 'EOF'
[{
  "origin": ["https://YOUR-NETLIFY-DOMAIN.netlify.app", "http://localhost:8080"],
  "method": ["PUT", "GET"],
  "responseHeader": ["Content-Type", "x-goog-resumable"],
  "maxAgeSeconds": 3600
}]
EOF
gsutil cors set /tmp/cors.json gs://autoencoder_data
```

### 7.3 Firestore

```bash
gcloud firestore databases create --project=autoencoders-census --location=us-central1
```

The `jobs` collection is created automatically when the first document is written.

### 7.4 Pub/Sub ✓ provisioned

```bash
# Topic: Express server publishes here when a file is uploaded
gcloud pubsub topics create job-upload-topic --project=autoencoders-census

# Subscription: Python worker listens here
gcloud pubsub subscriptions create job-upload-topic-sub \
  --topic=job-upload-topic \
  --project=autoencoders-census \
  --ack-deadline=600
```

Both the topic and subscription are now live in the `autoencoders-census` project.

### 7.5 Artifact Registry + Docker image (for Vertex AI mode only)

```bash
# Create the Docker repo
gcloud artifacts repositories create autoencoder-repo \
  --repository-format=docker \
  --location=us-central1 \
  --project=autoencoders-census

# Build and push the training container
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t us-central1-docker.pkg.dev/autoencoders-census/autoencoder-repo/trainer:v1 .
docker push us-central1-docker.pkg.dev/autoencoders-census/autoencoder-repo/trainer:v1
```

### 7.6 Service account

```bash
# Create service account for the Express server and Python worker
gcloud iam service-accounts create autoencoder-web \
  --display-name="AutoEncoder Web Service" \
  --project=autoencoders-census

SA=autoencoder-web@autoencoders-census.iam.gserviceaccount.com

# Grant required roles
gcloud projects add-iam-policy-binding autoencoders-census \
  --member="serviceAccount:$SA" --role="roles/storage.admin"
gcloud projects add-iam-policy-binding autoencoders-census \
  --member="serviceAccount:$SA" --role="roles/datastore.user"
gcloud projects add-iam-policy-binding autoencoders-census \
  --member="serviceAccount:$SA" --role="roles/pubsub.editor"
gcloud projects add-iam-policy-binding autoencoders-census \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding autoencoders-census \
  --member="serviceAccount:$SA" --role="roles/iam.serviceAccountUser"

# Download key for local/server use
gcloud iam service-accounts keys create frontend/service-account-key.json \
  --iam-account=$SA
```

### 7.7 Environment variables

**`frontend/.env`** (for Express API server):
```env
GCS_BUCKET_NAME=autoencoder_data
GOOGLE_CLOUD_PROJECT=autoencoders-census
PUBSUB_TOPIC_ID=job-upload-topic
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
```

**Python worker** (shell env or `.env` in repo root):
```env
GOOGLE_CLOUD_PROJECT=autoencoders-census
GCS_BUCKET_NAME=autoencoder_data
PUBSUB_SUBSCRIPTION_ID=job-upload-topic-sub
GOOGLE_APPLICATION_CREDENTIALS=frontend/service-account-key.json
```

### 7.8 Frontend deployment (Netlify)

The `frontend/netlify.toml` is already configured. In Netlify dashboard:

1. Connect the repo, set base directory to `frontend/`
2. Build command: `npm run build:client`
3. Publish directory: `dist/spa`
4. Set the same environment variables from 7.7 in the Netlify UI
5. The `/api/*` routes are redirected to Netlify Functions via `netlify.toml`

### 7.9 Worker hosting

The Python worker (`worker.py`) is a long-running Pub/Sub subscriber. Options:

| Option | Command | Best for |
|--------|---------|----------|
| Local dev | `python worker.py` | Development |
| Local (Vertex AI) | `python worker.py --mode=vertex` | When you want GCP to do the ML |
| GCE VM | Deploy on `e2-small` with startup script | Persistent demo |
| Cloud Run (min-instances=1) | Containerize `worker.py` | Production |

### 7.10 Quick-start (minimal demo, no Vertex AI)

For the simplest possible demo that runs the full pipeline:

1. Complete steps 7.1-7.4 and 7.6-7.7 (skip 7.5 -- no Vertex AI needed)
2. Terminal 1: `python worker.py` (local mode, default)
3. Terminal 2: `cd frontend && npm install && npm run dev`
4. Open `http://localhost:8080`, upload a CSV, see results

## 8. Handle Missing Data Properly

### Rationale

The current approach converts all NaN values to the string `"missing"` (`main.py:195`, `fillna("missing")`) and then one-hot encodes it like any other category. This has two significant problems:

1. **Artificially low outlier scores for incomplete rows.** A respondent who left many questions blank will have most of their one-hot vectors match the "missing" category perfectly. The autoencoder learns to reconstruct "missing" easily (it's a common, consistent pattern), so these rows get *low* reconstruction error and are never flagged — even though a row with 80% missing answers is inherently suspicious.

2. **Wasted model capacity.** The autoencoder spends parameters learning to reconstruct "missing" tokens instead of learning the actual joint distribution of real survey responses. Every attribute's softmax output includes a "missing" class that competes with real categories, diluting the signal.

The principled fix: **mask missing attributes out of the loss** so the model only trains on observed values, and use the decoder's predictions on masked positions as imputation recommendations.

### Current state (what exists today)

| File | What it does | Status |
|------|-------------|--------|
| `main.py:195` | `fillna("missing")` — converts NaN to a categorical value | **Active**, the primary missing-data strategy |
| `main.py:526` | Same `fillna("missing")` in `find_outliers` command | **Active**, duplicated |
| `features/transform.py:93-121` | `add_missing_indicators()` — creates binary `MISSING__col` columns | **Defined but never called** |
| `model/autoencoder.py:81-96` | `masked_mse()` — MSE loss that masks NaN positions | **Defined but never used** (models use categorical crossentropy) |
| `evaluate/evaluator.py:20-47` | `impute_missing_values()` — iterative autoencoder imputation | **Defined but commented out** at line 108 |
| `model/autoencoder.py:78` | `split_train_test()` calls `.dropna()` | **Active**, but no-op since NaN was already filled |
| `dataset/loader.py:380,389` | `convert_to_categorical()` creates `"missing"` bin for NaN values | **Active** for numeric-to-categorical binning |

### 8.1 Change missing value encoding in `features/transform.py`

Instead of one-hot encoding `"missing"` as its own category, represent missing attributes as:
- **All-zeros** in the one-hot group for that attribute (no category is active)
- A separate **binary missing indicator** column per attribute (1 = missing, 0 = observed)

The `add_missing_indicators()` method already creates these indicator columns — wire it into `vectorize_table()`. Update the `fillna("missing")` calls in `main.py` to instead preserve NaN through vectorization, letting `Table2Vector` handle the encoding.

**Files**: `features/transform.py`, `main.py` (lines 195-196, 526-527)

### 8.2 Build a per-attribute mask tensor

Create a boolean mask tensor of shape `(batch_size, num_attributes)` where `mask[i][j] = True` if attribute `j` is observed for row `i`. This mask must travel alongside the data through:
- `vectorize_table()` → return `(vectorized_df, mask_df)` or embed the mask as columns
- `split_train_test()` → split the mask in parallel with the data
- `model.fit()` → pass the mask via `sample_weight` or a custom training loop

The mask is derived from the missing indicator columns created in 8.1. Each indicator column maps to one attribute group in the one-hot vector.

**Files**: `features/transform.py`, `model/autoencoder.py`, `model/variational_autoencoder.py`, `train/trainer.py`

### 8.3 Modify loss functions to skip masked attributes

Update the per-attribute loss computation in both:
- `model/loss.py` (`CustomCategoricalCrossentropyAE.call`) — the `for categories in self.attribute_cardinalities` loop already iterates per attribute. Add a mask parameter; for masked attributes, contribute zero to the loss. Divide by the number of *observed* attributes, not total attributes.
- `model/base.py` (`VAE.reconstruction_loss`) — same change for the VAE path.

The existing `masked_mse` in `autoencoder.py:81-96` shows the right idea (masking NaN positions), but operates at the element level rather than the attribute-group level. The new masking should zero out entire attribute groups.

**Files**: `model/loss.py`, `model/base.py`

### 8.4 Update outlier scoring to use only observed attributes

In `evaluate/outliers.py` (`get_outliers_list`):
- Compute per-attribute reconstruction error only for observed attributes
- Average over observed attributes (not all attributes) to get the row-level outlier score
- Add a `missingness_fraction` column: the fraction of attributes that were missing for each row. A row missing 80% of its values is suspicious regardless of reconstruction quality.

This gives two complementary signals: "how anomalous are the answers you *did* give?" and "how much did you skip?"

**Files**: `evaluate/outliers.py`, `model/base.py`

### 8.5 Surface imputation recommendations

The decoder naturally produces predictions for all attributes, including masked ones. For missing attributes, the decoder's softmax output represents the model's best guess given the observed values and the learned joint distribution.

- In `evaluate/outliers.py`, extract the decoder's predicted category for each missing attribute
- Add these as columns in the output (e.g., `imputed__Q5`, `imputed__Q12`)
- Optionally include confidence (max softmax probability) for each imputed value

Wire up the existing `impute_missing_values()` in `evaluate/evaluator.py` (currently commented out) or replace it with a simpler single-pass approach — the masking-based model should produce good imputations in one forward pass without iterative refinement.

**Files**: `evaluate/outliers.py`, `evaluate/evaluator.py`

### 8.6 (Stretch) Add denoising / input dropout during training

Train a denoising autoencoder: during training, randomly mask out some fraction (e.g., 10-20%) of *observed* attributes — set their one-hot to all-zeros and their missing indicator to 1 — then require the model to reconstruct them. This teaches the model to impute from partial observations and makes it robust to arbitrary missing-data patterns.

This is particularly valuable when real missing-data rates are low (model rarely sees missing inputs during training) but you still want good imputation at inference time.

**Files**: `train/trainer.py` or a new data augmentation utility

### 8.7 Update tests

- Update `tests/features/test_transform.py` to cover the new all-zeros + missing indicator encoding
- Add tests for masked loss computation (verify that masked attributes contribute zero to loss)
- Add tests for outlier scoring with partially-missing rows
- Verify that a fully-observed row produces the same score as before (backward compatibility)

**Files**: `tests/features/test_transform.py`, `tests/model/test_loss.py`, new test file for outlier scoring

## 9. Strategic Considerations and Approach-Level Gaps

The items above are concrete bugs and tasks. This section captures higher-level concerns about the project's methodology, product design, and scientific validity.

### 9.1 The Rule of 9 discards most of the signal in many datasets

The Rule of 9 drops every column with more than 9 unique values or only 1 unique value. For many real-world surveys this eliminates:
- **All numeric columns** (age, income, Likert scales coded as integers, slider responses)
- **All free-text and high-cardinality fields** (occupation, zip code, open-ended responses)
- **All ordinal variables** with more than 9 levels (e.g., 1-10 satisfaction scales)

What remains is only low-cardinality categoricals — often demographic questions (gender, education bracket, region) rather than the substantive survey items where inattentive responding is most visible. The model may end up detecting demographic outliers (unusual demographic combinations) rather than inattentive responders.

Additionally, there is a gap in the binning logic: `convert_to_categorical` in `dataset/loader.py` only bins columns with >20 unique values or float64 type. Integer columns with 10-20 unique values (e.g., a 1-10 satisfaction scale) are neither binned nor pass the Rule of 9 — they are silently dropped.

One-hot encoding also discards ordinality: "Strongly Agree" vs "Agree" contributes the same reconstruction error as "Strongly Agree" vs "Strongly Disagree". A respondent who is slightly off contributes the same error as one who is maximally wrong.

**Recommendation**: Support ordinal and numeric variables natively rather than discarding them. Numeric columns could use MSE reconstruction loss directly; ordinal columns could use an ordered representation or ordinal distance-weighted error. At minimum, report to the user exactly which columns were kept and which were dropped, so they can judge whether the analysis is meaningful for their data.

### 9.2 No comparison to traditional survey quality methods

Survey methodology has well-established inattentive responding detection methods that predate autoencoders:
- **Longstring analysis**: detecting straight-lining (same answer repeated across many questions)
- **Response time**: flagging respondents who complete the survey unrealistically fast
- **Intra-individual response variability (IRV)**: measuring whether a respondent's answers show enough variance
- **Mahalanobis distance**: classical multivariate outlier detection
- **Even-odd consistency**: checking whether answers to semantically similar items are consistent
- **Attention check items**: embedding known-answer questions ("select strongly agree for this item")

The autoencoder approach is interesting because it can detect multivariate patterns that simpler methods miss. But without benchmarking against these baselines, it is hard to know whether the added complexity is justified. Some of these methods (longstring, IRV) are trivial to compute and might catch 80% of the bad respondents that the autoencoder catches.

Note that some of the built-in datasets (Pennycook, bot_bot_mturk) include timing data (`time_diff`, `submit_diff` columns), but these are likely dropped by the Rule of 9 because they have too many unique values. The autoencoder never sees the single strongest signal of low-quality responding.

**Recommendation**: The Chow-Liu tree in `chow_liu_rank.py` (section 10) already provides a principled probabilistic baseline. Once it is integrated into the CLI (task 10.1), benchmark it against the AE using `evaluate_on_condition`. Consider also implementing longstring analysis and Mahalanobis distance. The most impactful improvement may be an **ensemble approach** that combines the autoencoder's reconstruction error with the Chow-Liu log-likelihood and traditional indicators (completion time, straightlining score, attention check failures) into a composite score.

### 9.3 Evaluation relies on proxy ground truth, not validated labels

The `evaluate_on_condition` command flags respondents who fail attention check items and then measures whether the autoencoder also flags them. But attention check failures are just one kind of bad responding. The autoencoder might detect different (also valid) patterns of inattention that attention checks miss, or it might be detecting something else entirely (e.g., demographic minorities).

There is no evaluation that disentangles:
- True inattentive responders (random clicking)
- Mischievous responders (deliberately wrong answers)
- Unusual but genuine respondents (people with uncommon but real demographic/opinion combinations)
- Data entry errors (typos, miscoded values)

There is also a circularity concern with the SADC evaluation: `find_outlier_data_sadc_2017` in `dataset/loader.py` constructs ground truth by checking whether respondents simultaneously report extreme values across multiple food/body measurement variables. This heuristic is itself an outlier detection rule — the autoencoder is being benchmarked on whether it can replicate a simpler, interpretable rule. The autoencoder's value proposition should be that it discovers patterns that hand-crafted rules cannot, but the evaluation framework cannot measure that.

**Recommendation**: Include synthetic contamination experiments — inject known-bad rows (random responses, straight-lined responses, copy-pasted rows, bot-like patterns) into clean data and measure detection rates by type and severity. This gives clean, unambiguous ground truth and lets you characterize exactly what kinds of data quality issues the autoencoder can and cannot detect.

### 9.4 Outlier score lacks calibration and interpretability

The reconstruction error is a raw number with no natural scale. Users see "this row has error 0.73" but have no way to know:
- Is 0.73 high or low?
- What percentile does this correspond to?
- How confident should I be that this is a real outlier versus natural variation?

The current UI shows a ranked list, which is better than raw scores, but still doesn't help users decide where to draw the threshold — how many respondents should they actually remove?

**Recommendation**:
- Report percentile ranks rather than raw scores ("this respondent is in the top 3% most anomalous")
- Provide a suggested threshold based on the distribution of errors (e.g., the elbow point, or a statistically-derived cutoff)
- Show the distribution of reconstruction errors (histogram) so users can visually identify the outlier tail

### 9.5 The model trains on the full dataset including outliers

The autoencoder is trained on the complete dataset, including the inattentive respondents it is supposed to detect. If the dataset has a high contamination rate (e.g., 20% bad respondents), the autoencoder learns to partially reconstruct the bad patterns, reducing their reconstruction error and making them harder to detect.

This is a well-known limitation of autoencoders for outlier detection — they work best when the contamination rate is low (< 5%).

**Recommendation**: Consider iterative training — train once, remove the worst outliers, retrain on the cleaned data, re-score. This is analogous to robust estimation in statistics. Alternatively, use the percentile-trimmed loss (which already exists in `CustomCategoricalCrossentropyAE`) more aggressively to down-weight high-error samples during training.

### 9.6 Product does not help users act on results

The current product answers "which rows are anomalous?" but does not help users with the next steps:
- Should they delete these rows, impute them, or just flag them?
- How does removing outliers change their substantive analysis results?
- Is the outlier a data quality problem or a genuinely unusual respondent worth keeping?

For a survey researcher, the most valuable output would be: "here is your dataset with and without the flagged rows, and here is how your key findings change." This connects the technical detection to the practical decision.

**Recommendation**: Consider adding a "sensitivity analysis" mode that shows how basic summary statistics (means, correlations, group comparisons) change when flagged respondents are excluded. This is what survey researchers actually need to decide whether data cleaning matters for their conclusions.

### 9.7 No support for panel/longitudinal data or respondent-level modeling

The model treats each row independently. For panel surveys (same respondents measured multiple times), the system cannot detect respondents who are consistent *across* waves in a suspicious way (e.g., copying their previous responses) or who show impossible changes between waves.

### 9.8 The web UI and CLI are separate products with different capabilities

The CLI supports training, evaluation, hyperparameter search, generation, PCA baselines, and conditional evaluation. The web UI only supports upload, train, and score. A user who starts with the web UI cannot access any of the advanced CLI features.

This creates a fragmented experience — power users must switch to the CLI, casual users cannot access most functionality. Consider which features from the CLI should be exposed in the web UI, or whether the CLI should be the primary interface with the web UI serving as a simple demo.

### 9.9 VAE adds complexity but unclear value for outlier detection (LOW PRIORITY)

The VAE is not actively used. The focus is on the standard AE and the Chow-Liu tree baseline. The VAE code remains in the codebase but should not receive further investment unless benchmarks (task 3.4) show it adds clear value over the AE for outlier detection.

### 9.10 Default latent space dimension is too small

The default configs (`config/simple_autoencoder.yaml`, `config/simple_variational_autoencoder.yaml`) use `latent_space_dim: 2`. For survey data with potentially 50+ categorical attributes, a 2-dimensional bottleneck is extremely aggressive — the model can only capture the two most dominant patterns of variation. This may catch the grossest outliers (random responders) but will miss subtler forms of data quality issues. The hyperparameter search configs allow up to 16 dimensions, which is more reasonable.

**Recommendation**: Increase the default to at least 8-16 for production use. Reserve `latent_space_dim: 2` for visualization and exploratory analysis.

### 9.11 Survey skip patterns create false positives

Real-world surveys routinely use skip/branching logic: "If you answered Yes to Q12, answer Q13-Q20; otherwise skip to Q21." Respondents who legitimately skipped a block of questions will have many missing values that all perfectly reconstruct (Section 8's problem), or the model sees their "not applicable" pattern as anomalous because the subgroup that took this branch is smaller. The system has no way to represent "this question was not applicable to this respondent" versus "this respondent skipped a question they should have answered."

### 9.12 No minimum sample size guidance

The autoencoder needs enough data to learn meaningful patterns. With default architecture (128-unit layers) and typical survey dimensionality (50 attributes, ~200 one-hot dimensions), the model has substantial parameters relative to typical survey sample sizes. A survey with 200 respondents will not train a useful autoencoder. The project provides no guidance on minimum sample sizes, which means web UI users with small pilot surveys will get meaningless results with no warning.

**Recommendation**: Add a pre-training check that warns users when the dataset is too small relative to the model capacity (e.g., fewer rows than one-hot dimensions, or fewer than 500 rows).

### 9.13 GCP-coupled architecture limits adoption

The web frontend requires Google Cloud Storage, Pub/Sub, and Firestore — meaning even evaluating the web product requires a GCP project with billing enabled. For academic adoption, this is prohibitive. For typical survey datasets (1,000-10,000 rows, 50-100 columns), local processing completes in seconds to minutes, making the distributed architecture unnecessary.

**Recommendation**: Add a fully local web mode that processes files in-memory on a single server without any cloud dependencies. The `--mode=local` worker is a step in this direction but still requires GCS for file upload and Firestore for job tracking.

## 10. Chow-Liu Tree Outlier Scoring

### Overview

`chow_liu_rank.py` adds a Chow-Liu tree-based outlier scoring method. It fits a maximum spanning tree of pairwise mutual information on categorical data and computes per-row log-likelihood — rows with low log-likelihood are outliers. This is a fast, non-neural baseline with no training hyperparameters.

### ~~10.1 Clean up `chow_liu_rank.py`~~ DONE
Moved `DataLoader` and `define_necessary_elements` imports from module level into `if __name__ == "__main__"` block so the module is a clean, standalone library. The module now only depends on numpy, pandas, typing, and dataclasses. Fixed the `__main__` block to use `error = 1 - pct` (higher = more anomalous) instead of using `pct` directly as error (which was inverted).

### ~~10.2 Integrate Chow-Liu as a CLI command~~ DONE
Added `chow_liu_outliers` CLI command in `main.py` that:
- Uses the same `define_necessary_elements()` → `DataLoader` → `load_data()` pipeline as all other commands
- Calls `prepare_for_categorical()` (new helper extracted from `prepare_for_model()`) for cleaning: fillna("missing") → astype(str) → Rule-of-9 — no vectorization needed since CLTree operates on raw categorical data
- Fits CLTree via `rank_rows_by_chow_liu()` with configurable `--alpha` (Laplace smoothing) and `--mi_subsample` options
- Outputs `errors.csv` with `error = 1 - pct` column, compatible with `evaluate_on_condition`
- Logs the top-5 strongest tree edges (highest mutual information) for interpretability

Also extracted `prepare_for_categorical()` from `prepare_for_model()` to avoid duplicating the cleaning logic. `prepare_for_model()` now calls `prepare_for_categorical()` internally for its first two steps.

### ~~10.3 Add Chow-Liu to the model factory (optional)~~ DONE (decided against)
Decision: Keep as a separate CLI command (`chow_liu_outliers`). The tree has a fundamentally different API (no `fit`/`predict` in the Keras sense, no config YAML, no vectorization step), so forcing it through the autoencoder pipeline would add complexity without benefit.

### ~~10.4 Add tests for Chow-Liu tree~~ DONE
Added `tests/test_chow_liu.py` with 26 tests across 4 test classes:
- **`TestCLTreeFit`** (9 tests): schema building, tree structure (correct number of edges, parent/child relationships), root selection, marginal distributions sum to 1, CPT rows sum to 1, explicit root, mi_subsample
- **`TestCLTreeLogLikelihood`** (4 tests): correct output length, all values negative, no NaN/inf, training data scores higher than random data
- **`TestRankRowsByChowLiu`** (9 tests): scoring columns present, original columns preserved, rank is valid permutation, pct/gmean_prob in [0,1], z-scores centered at 0, logp/avg_logp consistency
- **`TestChowLiuOutlierDetection`** (4 tests): injected random rows score lower than correlated rows, error column compatible with evaluate_on_condition, NaN handling, single-column edge case

### ~~10.5 Benchmark Chow-Liu vs AE for outlier detection~~ DONE

Ran comparisons on SADC 2015 and SADC 2017 using the composite outlier indicator (respondents with 3+ simultaneous extreme food/body-measurement values) as ground truth. Benchmark script: `benchmark_ae_vs_cl.py`. Config for tuned AE: `config/benchmark_autoencoder.yaml` (latent_dim=8, 50 epochs, lr=0.001).

#### Results: SADC 2017 (14,765 rows, 230 outliers, 1.56% prevalence)

| Metric | AE (latent=8, 50ep) | AE (latent=2, 5ep) | Chow-Liu (α=1.0) |
|--------|--------------------:|-------------------:|------------------:|
| ROC AUC | **0.7568** | 0.7420 | 0.7517 |
| Avg Precision | **0.0629** | 0.0486 | 0.0501 |
| Precision@10 | **0.4000** | 0.1000 | 0.0000 |
| Precision@50 | **0.2200** | 0.1800 | 0.0400 |
| Precision@100 | 0.1100 | **0.1200** | 0.0900 |
| Lift@10 | **25.7x** | 6.4x | 0.0x |
| Recall@500 | **0.1522** | 0.1217 | **0.1522** |

#### Results: SADC 2015 (15,624 rows, 295 outliers, 1.89% prevalence)

| Metric | AE (latent=8, 50ep) | Chow-Liu (α=1.0) |
|--------|--------------------:|------------------:|
| ROC AUC | **0.7176** | 0.7060 |
| Avg Precision | 0.0446 | **0.0466** |
| Precision@100 | 0.0000 | **0.0200** |
| Precision@295 (=n_outliers) | 0.0373 | **0.0712** |
| Recall@500 | 0.0949 | **0.1288** |

#### Key findings

1. **AE and Chow-Liu have comparable ROC AUC** (~0.71-0.76 on both datasets). Neither method strongly dominates.
2. **AE has better top-of-list precision on SADC 2017** — the AE concentrates true outliers in its top-10 and top-50 predictions much better than Chow-Liu (precision@10 of 0.40 vs 0.00). This suggests the AE captures multivariate patterns that the tree misses for the most extreme cases.
3. **Chow-Liu is more robust on SADC 2015** — the AE's top-k precision drops to near zero on SADC 2015, while Chow-Liu maintains modest but consistent performance. This suggests the AE may overfit to dataset-specific patterns.
4. **Larger latent dimension helps** — the tuned AE (latent=8, 50 epochs) outperforms the default config (latent=2, 5 epochs) on AUC and average precision, confirming TASKS.md 9.10's recommendation to increase the default.
5. **Both methods struggle with low-prevalence detection** — with only 1.6-1.9% outlier prevalence, even the best method achieves only ~6% average precision. This highlights the challenge noted in 9.5 (training on contaminated data) and 9.3 (proxy ground truth).
6. **Chow-Liu is dramatically faster** — fits in <1 second vs. ~50 seconds for AE training. For quick screening, Chow-Liu is a practical choice.
7. **Ensemble potential** — since AE excels at top-of-list and Chow-Liu at consistency, combining both scores (task 9.2's recommendation) could yield the best of both worlds.
