# TASKS.md -- Product Launch for Survey Outlier Detection

## 1. Stabilize the Core Pipeline

### 1.1 Unify data-loading return format
The `DataLoader` methods return inconsistent formats -- some return `(df, variable_types)`, others return `(df, metadata_dict)`. The `train` and `find_outliers` commands in `main.py` work around this with nested isinstance checks (lines 174-192, 511-526). Refactor `DataLoader.prepare_original_dataset()` and all `load_*` methods to return a consistent `(DataFrame, dict)` where the dict always has `"variable_types"` and `"ignored_columns"` keys.

### 1.2 Extract shared data-cleaning logic
The data-cleaning pipeline (fillna, astype(str), Rule of 9 filter, vectorization, float32 conversion) is duplicated between the `train` command (main.py lines 196-253), the `find_outliers` command (main.py lines 529-570), and `run_training_pipeline()` (main.py lines 57-91). Extract this into a single reusable function (e.g., `prepare_for_model(df) -> (vectorized_df, variable_types, cardinalities)`) and call it from all three places.

### 1.3 Consolidate dataset config definitions
Dataset-specific column configs (`drop_columns`, `rename_columns`, `interest_columns`) are duplicated across `main.py` (in `train`, `find_outliers`, `evaluate_on_condition`, `pca_baseline`) and `utils.py` (`define_necessary_elements`). Move all dataset configs into a single YAML or Python registry to eliminate drift between copies.

### 1.4 Fix broken test_loader.py tests
`tests/dataset/test_loader.py` references `DataLoader.DATASET_URL_2015` and `DataLoader.DATASET_URL_2017` class attributes and calls `DataLoader()` with no arguments. These no longer match the current `DataLoader.__init__` signature which requires `drop_columns`, `rename_columns`, and `columns_of_interest`. Update the tests to match the current API.

### 1.5 Fix broken test_loss.py tests
`tests/model/test_loss.py` imports `CustomCategoricalCrossentropyVAE` which does not exist in `model/loss.py`. All tests in this file fail with `ImportError`. Fix the import and update tests to match the current loss API.

### 1.6 Fix CLI commands that skip data cleaning
Several CLI commands bypass the `fillna` / `astype(str)` / Rule-of-9 pipeline that `train` and `find_outliers` apply, meaning data seen during evaluation is preprocessed differently from training:
- `search_hyperparameters` (main.py:334-340)
- `evaluate` (main.py:411-412)
- `evaluate_on_condition` (main.py:1056-1073)
- `pca_baseline` (main.py:1592 — also crashes on `None.split(",")`)

Additionally, these commands use `describe().T["unique"]` to get cardinalities, which raises `KeyError` on numeric columns. The `train`/`find_outliers` commands correctly use `nunique()` instead.

### 1.7 Fix dataset config inconsistencies across duplicated blocks
The column configs in `evaluate_on_condition` / `pca_baseline` (main.py) differ from `define_necessary_elements` (utils.py) for almost every dataset:
- `moral_data`: `range(12, 78)` vs `range(12, 77)`
- `public_opinion`: `range(21, 176)` vs `range(21, 175)`
- `mturk_ethics`: includes `+ [107, 108]` vs omits them
- `racial_data`: includes `+ [74,75,76]` vs omits them
- `bot_bot_mturk`: `range(20, 35)` vs `range(20, 34)` (and 35 appears twice in the main.py version)
- `inattentive`: includes `3` vs omits it
- `attention_check`: includes `+ [2]` vs omits it

These produce silently different results depending on which CLI command is used. Blocked by 1.3 (consolidate configs).

### 1.8 Fix `COLUMNS_OF_INTEREST` integer-vs-name mismatch
`COLUMNS_OF_INTEREST` is set as a list of integer indices in all dataset configs, but `DataLoader.load_original_data()` (loader.py:333-336) filters with `if c in original_df.columns`, comparing integers to string column names. The filter never matches, producing an empty DataFrame. The logic should use `iloc` for integer indices or convert configs to column names.

### 1.9 Fix `run_training_pipeline` typo
Default prior is `"guassian"` (main.py:57) but the VAE checks `if prior == "gaussian"`. The misspelling causes the function to always raise `ValueError("Invalid prior")`.

### 1.10 Fix `generate` command
`generate` (main.py:656-657) accesses `model.get_config()["prior_means"]` and `["prior_log_vars"]`, but the VAE's `get_config()` does not include these keys. The command crashes with `KeyError`.

### 1.11 Fix `CustomCategoricalCrossentropyAE.get_config` missing `percentile`
`model/loss.py:143-144` omits the `percentile` parameter from `get_config()`. When a model is saved and reloaded, the percentile reverts to the default (80) regardless of the value used during training.

### ~~1.12 Remove DEBUG print statements~~ DONE
All `print("DEBUG: ...")` statements in `main.py` have been replaced with proper `logger.debug()`, `logger.warning()`, and `logger.error()` calls. Emoji-prefixed error prints were also cleaned up.

## 2. Harden the Upload Pipeline (Web UI)

### ~~2.1 Implement end-to-end upload processing in worker.py~~ DONE
`worker.py` now has two modes: `--mode=local` (default) processes uploads entirely locally (download from GCS, train autoencoder, score outliers, write to Firestore), and `--mode=vertex` dispatches to Vertex AI. The local mode enables running the demo without any Vertex AI setup. The worker also writes a `"processing"` status to Firestore before starting, so the frontend can show progress.

### 2.2 Handle arbitrary user CSV uploads robustly
The current `load_uploaded_csv` path in `DataLoader` calls `prepare_original_dataset()` which bins numeric variables and applies the Rule of 9 filter. Verify this works correctly for CSVs with:
- Mixed numeric and categorical columns
- Columns with special characters, spaces, or unicode in names
- Very wide datasets (100+ columns)
- Datasets with mostly missing values
- Completely numeric datasets (no categorical columns to keep)

Add integration tests covering these edge cases.

### 2.3 Add input validation and error reporting
When a user uploads a malformed CSV (wrong encoding, not actually CSV, empty file, single-column file), the system should return a clear error message to the frontend rather than a stack trace. Add validation in the upload path and propagate structured error messages to Firestore/frontend.

### 2.4 Add API authentication
The Express API endpoints are unauthenticated (noted in README as a current limitation). Add at minimum API key authentication or session-based auth before deploying to any network accessible by others.

### 2.5 Fix security issues in the upload/API pipeline
Several security issues need to be addressed before any public deployment:
- **Open CORS**: `cors()` with no arguments (frontend/server/index.ts:45) allows any origin. Restrict to the frontend's actual domain.
- **Path traversal via filename**: `req.file.originalname` is user-controlled and used directly in the GCS path (frontend/server/index.ts:64). Sanitize by stripping directory separators and using only the basename.
- **No file-type validation on upload**: The multer upload accepts any file type. Add a `fileFilter` to restrict to `.csv` / `text/csv`.
- **No rate limiting**: Neither the upload nor polling endpoints have rate limiting. An attacker could fill the GCS bucket or abuse the polling endpoint.
- **Signed URL endpoint unauthenticated**: `/upload-url` in both `routes/jobs.ts` and `routes/demo.ts` generates GCS write URLs for anyone. The demo route also lacks any try/catch (crashes the server on error).
- **Job IDs passed to Firestore without validation**: User-supplied `jobId` and route params are passed directly to `firestore.collection("jobs").doc(id)`.

### 2.6 Fix worker reliability issues
- **Ack race condition** (worker.py:219-232): The message is acked only after processing completes. If processing exceeds the Pub/Sub ack deadline, the message is redelivered and the entire pipeline runs again. Extend the ack deadline during processing or ack earlier and track state in Firestore.
- **No message field validation** (worker.py:223-225): If `jobId`, `bucket`, or `file` is missing from the message, `None` is passed downstream, causing a crash deep in GCS code with an unhelpful error.
- **No env var validation at startup** (worker.py:41-43): If `GOOGLE_CLOUD_PROJECT`, `GCS_BUCKET_NAME`, or `PUBSUB_SUBSCRIPTION_ID` is unset, the worker starts but crashes on the first message. Fail fast at startup.

### 2.7 Remove hardcoded GCP identifiers from source code
- `worker.py:196`: Hardcoded service account email `203111407489-compute@developer.gserviceaccount.com`
- `worker.py:181`: Hardcoded staging bucket `gs://autoencoders-census-staging`
- `train/task.py:32`: Hardcoded project ID `autoencoders-census` instead of reading from `os.getenv("GOOGLE_CLOUD_PROJECT")`

These should all be read from environment variables.

### 2.8 Fix inconsistent outlier scoring between CLI and web
The CLI `find_outliers` command uses `VAE.reconstruction_loss()` (per-attribute categorical crossentropy normalized by log-cardinality), but the worker (worker.py:141) and Vertex AI task (train/task.py:94) use raw MSE on one-hot vectors: `np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)`. These are fundamentally different scoring functions. Outlier rankings from the web UI differ significantly from CLI results, even on the same data and model. Unify on a single scoring function.

## 3. Improve Model Quality and Configurability

### 3.1 Make the "Rule of 9" threshold configurable
The max-unique-values threshold of 9 is hardcoded in `DataLoader.prepare_original_dataset()` (line 439) and duplicated in `main.py` `train`/`find_outliers` commands. Make this a CLI parameter and config option so users can adjust it for their specific datasets.

### 3.2 Support custom model configs for uploaded data
When a user uploads a CSV through the web UI, the system should either auto-select reasonable hyperparameters based on the data shape or allow the user to choose a config preset. Currently, the uploaded data path does not specify which config to use.

### 3.3 Add per-column outlier contribution scores
The `get_outliers_list()` function returns an aggregate reconstruction error per row. For interpretability, also compute and return per-column reconstruction error so users can see *which* survey questions a flagged respondent answered anomalously.

### 3.4 Benchmark model variants
Run systematic comparisons of AE vs. VAE (Gaussian) vs. VAE (Gumbel) vs. PCA baseline on all built-in datasets. Record metrics (accuracy, lift, ROC AUC from `evaluate` command, plus outlier detection precision from `evaluate_on_condition`) and document which approach works best under what conditions.

### 3.5 Fix numerical stability issues in loss computation
- **Division by log(1) = 0**: `model/loss.py:62` and `model/base.py:121` normalize per-attribute crossentropy by `np.log(categories)`. If any attribute has cardinality 1, this is division by zero, producing `inf`/`NaN` loss. Commands that skip the Rule-of-9 filter (1.6) are vulnerable. Guard with `max(np.log(categories), epsilon)`.
- **`Concatenate()` with single attribute**: `model/autoencoder.py:252` and `model/layers.py:39` call `Concatenate()(decoded_attrs)` which requires 2+ inputs. A single-column dataset passing Rule-of-9 crashes here.

### 3.6 Remove global eager mode
`tf.config.run_functions_eagerly(True)` is set at module level in `model/base.py:8` and `model/variational_autoencoder.py:13`. This forces all TensorFlow functions in the process to run eagerly, causing 2-10x training slowdown. This was likely added for debugging. Remove it or guard behind a `DEBUG` environment variable.

### 3.7 Fix VAE serialization issues
- `kl_loss_weight` is overwritten to `0.0` in the constructor (model/base.py:36-40) for warmup scheduling. But `get_config()` (model/base.py:269-270) calls `.numpy()` on `self.kl_loss_weight`, which is a Python float (not a tensor) before the first training step, raising `AttributeError`.
- After deserialization via `from_config`, the warmup schedule restarts from 0 regardless of where training left off.

### 3.8 Fix data leakage in feature transformation
- **MinMaxScaler fit on train+test** (features/transform.py:65-70): The scaler is `fit_transform`'d on the entire DataFrame before train/test split. Test set min/max values leak into training normalization.
- **OneHotEncoder fit on full data** (features/transform.py:77): Same issue — categories present only in the test set are known to the encoder during training.
- **Index misalignment during one-hot concat** (features/transform.py:76-88): `df_encoded` gets a default RangeIndex, but `vectorized_df` may have a non-standard index (e.g., after `train_test_split`). The `pd.concat(..., axis=1)` aligns on index, and mismatched indices silently introduce `NaN` values. This is a data corruption bug.

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

### 5.1 Get all existing tests passing
Run `python -m pytest tests/ -v` and fix all failures. Known issues: `test_loader.py` has stale API references (1.4), `test_loss.py` imports a nonexistent class (1.5), and tests may need the SADC CSV data files present in `data/`.

### 5.2 Add integration tests for the CLI pipeline
Write tests that exercise the full `train -> find_outliers` pipeline on a small synthetic dataset (no GCP dependency). Verify that the output `errors.csv` is correctly sorted by reconstruction error.

### 5.3 Add tests for the web upload path
Write tests for `DataLoader.load_uploaded_csv()` with various CSV payloads (valid, empty, malformed, unicode). Mock GCS and Firestore interactions to test the worker pipeline end-to-end.

### 5.4 Set up CI pipeline
Add a GitHub Actions workflow that runs linting and tests on each push. Pin TensorFlow and other heavy dependencies to avoid CI build failures.

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

### 7.4 Pub/Sub

```bash
# Topic: Express server publishes here when a file is uploaded
gcloud pubsub topics create job-upload-topic --project=autoencoders-census

# Subscription: Python worker listens here
gcloud pubsub subscriptions create job-upload-topic-sub \
  --topic=job-upload-topic \
  --project=autoencoders-census \
  --ack-deadline=600
```

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

**Recommendation**: Implement at least longstring analysis and Mahalanobis distance as baseline comparisons. The `evaluate_on_condition` command already compares against ground-truth labels — use it to measure whether the autoencoder adds value over these simpler methods. The most impactful improvement may be an **ensemble approach** that combines the autoencoder's reconstruction error with traditional indicators (completion time, straightlining score, attention check failures) into a composite score. Each method has different blind spots; combining them would produce a far more robust quality indicator.

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

### 9.9 VAE adds complexity but unclear value for outlier detection

The VAE (Variational Autoencoder) variant adds significant complexity (Gumbel-Softmax prior, KL warmup scheduling, temperature annealing) but the project has no documented evidence that it outperforms the simpler AE for outlier detection.

For the outlier detection use case, the VAE's generative capabilities (sampling, latent interpolation) are not directly useful — what matters is reconstruction error quality. The VAE's KL regularization may actually *hurt* outlier detection by forcing the encoder to use a smooth latent space that compresses outlier information.

**Recommendation**: Run the benchmarks in task 3.4 before investing further in VAE improvements. If the simpler AE performs comparably, consider making VAE an optional advanced feature rather than a co-equal path.

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
