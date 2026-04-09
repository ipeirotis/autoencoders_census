# External Integrations

**Analysis Date:** 2026-01-23

## APIs & External Services

**Google Cloud Storage (GCS):**
- Purpose: CSV file upload/download for training data
- SDK/Client: `@google-cloud/storage` ^7.18.0 (Node), `google-cloud-storage==3.7.0` (Python)
- Auth: Service account key in `frontend/service-account-key.json`
- Implementation: `frontend/server/index.ts`, `frontend/server/routes/jobs.ts`, `worker.py`
- Features: Signed URL generation for direct browser uploads

**Google Cloud Firestore:**
- Purpose: Document database for job metadata and results
- SDK/Client: `@google-cloud/firestore` ^8.0.0 (Node), `google-cloud-firestore==2.22.0` (Python)
- Implementation: `frontend/server/index.ts`, `worker.py`, `train/task.py`
- Collections: `jobs` (stores job ID, status, results, timestamps)

**Google Cloud Pub/Sub:**
- Purpose: Asynchronous message queue for job dispatch
- SDK/Client: `@google-cloud/pubsub` ^5.2.0 (Node), `google-cloud-pubsub==2.34.0` (Python)
- Implementation: `frontend/server/routes/jobs.ts`, `worker.py`
- Topic: `job-upload-topic`
- Subscription: `job-upload-topic-sub` (configured via `PUBSUB_SUBSCRIPTION_ID`)

**Google Vertex AI:**
- Purpose: Managed ML training platform
- SDK/Client: `google-cloud-aiplatform==1.132.0`
- Implementation: `worker.py` (lines 54-61)
- Job Type: CustomContainerTrainingJob
- Container: `us-central1-docker.pkg.dev/{PROJECT_ID}/autoencoder-repo/trainer:v1`
- Machine Type: `n1-standard-4`

## Data Storage

**Databases:**
- Firestore - Document store for job metadata
  - Connection: Service account credentials
  - Client: Firestore SDK initialized in `frontend/server/index.ts`

**File Storage:**
- Google Cloud Storage - CSV uploads and model artifacts
  - Bucket: Configured via `GCS_BUCKET_NAME` env var
  - Auth: Service account key

**Caching:**
- Local filesystem - `cache/` directory for trained models
  - `cache/simple_model/` - Saved autoencoder weights

## Authentication & Identity

**Auth Provider:**
- Google Cloud IAM - Service account authentication
- Implementation: Service account JSON key files
- Token storage: Environment variables and key files

**Service Accounts:**
- Frontend: `frontend/service-account-key.json`
- Vertex AI: `203111407489-compute@developer.gserviceaccount.com`

## Monitoring & Observability

**Error Tracking:**
- Not configured (uses console logging)

**Analytics:**
- Not configured

**Logs:**
- Console/stdout logging via Python `logging` module
- Express server logs to stdout

## CI/CD & Deployment

**Hosting:**
- Vertex AI for ML training jobs
- Local development via Vite dev server

**CI Pipeline:**
- Not configured

**Container Registry:**
- Google Artifact Registry: `us-central1-docker.pkg.dev`
- Image: `autoencoder-repo/trainer:v1`

## Environment Configuration

**Development:**
- Required env vars: `GOOGLE_CLOUD_PROJECT`, `GCS_BUCKET_NAME`
- Secrets location: `.env` file (gitignored should be applied)
- Service account: Local key files

**Production:**
- Secrets management: GCP service account credentials
- Environment: Vertex AI managed environment

## Webhooks & Callbacks

**Incoming:**
- None configured

**Outgoing:**
- Pub/Sub message publishing to `job-upload-topic`
  - Trigger: POST `/api/jobs/start-job`
  - Payload: `{jobId, bucketName, filePath}`

## Data Pipeline Workflow

1. **Upload**: Frontend → Signed URL → GCS (direct browser upload)
2. **Trigger**: Express → Firestore doc → Pub/Sub message
3. **Process**: Worker → Vertex AI CustomContainerTrainingJob
4. **Train**: Container → Download CSV → Train model → Calculate errors
5. **Store**: Results → Firestore `jobs/{jobId}`
6. **Poll**: Frontend → GET `/api/jobs/job-status/:id` → Display results

---

*Integration audit: 2026-01-23*
*Update when adding/removing external services*
