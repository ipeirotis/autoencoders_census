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

### ~~1.5 Remove DEBUG print statements~~ DONE
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

## 3. Improve Model Quality and Configurability

### 3.1 Make the "Rule of 9" threshold configurable
The max-unique-values threshold of 9 is hardcoded in `DataLoader.prepare_original_dataset()` (line 439) and duplicated in `main.py` `train`/`find_outliers` commands. Make this a CLI parameter and config option so users can adjust it for their specific datasets.

### 3.2 Support custom model configs for uploaded data
When a user uploads a CSV through the web UI, the system should either auto-select reasonable hyperparameters based on the data shape or allow the user to choose a config preset. Currently, the uploaded data path does not specify which config to use.

### 3.3 Add per-column outlier contribution scores
The `get_outliers_list()` function returns an aggregate reconstruction error per row. For interpretability, also compute and return per-column reconstruction error so users can see *which* survey questions a flagged respondent answered anomalously.

### 3.4 Benchmark model variants
Run systematic comparisons of AE vs. VAE (Gaussian) vs. VAE (Gumbel) vs. PCA baseline on all built-in datasets. Record metrics (accuracy, lift, ROC AUC from `evaluate` command, plus outlier detection precision from `evaluate_on_condition`) and document which approach works best under what conditions.

## 4. Frontend Polish

### 4.1 Display per-column outlier explanations
When the results table shows flagged outliers, include a breakdown of which columns contributed most to the high reconstruction error (depends on task 3.3).

### 4.2 Add progress indicator for long-running jobs
Vertex AI jobs take 10-15 minutes. The frontend polls Firestore for completion. Add a progress state (e.g., "Preprocessing", "Training", "Scoring") that the worker writes to Firestore so users see meaningful status updates instead of just "Processing...".

### 4.3 Allow downloading results as CSV
Add an export button on the results page that lets users download the outlier-scored data as a CSV file.

### 4.4 Support job cancellation
Users currently cannot cancel running jobs from the UI (noted in README). Add a cancellation endpoint that marks the job as cancelled in Firestore and, if using Vertex AI, cancels the cloud job.

## 5. Testing and CI

### 5.1 Get all existing tests passing
Run `python -m pytest tests/ -v` and fix all failures. Known issues: `test_loader.py` has stale API references, and tests may need the SADC CSV data files present in `data/`.

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
