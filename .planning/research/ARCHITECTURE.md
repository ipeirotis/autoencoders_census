# Architecture Research: Production Features Integration

**Domain:** ML-as-a-Service Web Platform (AutoEncoder Outlier Detection)
**Researched:** 2026-03-24
**Confidence:** HIGH

## Existing Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REACT FRONTEND (Vite)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Index.tsx    │  │ Dropzone     │  │ ResultCard   │              │
│  │ (Main UI)    │  │ (Upload UI)  │  │ (Display)    │              │
│  └──────┬───────┘  └──────────────┘  └──────────────┘              │
│         │                                                            │
│         │ uploadCsv(), checkJobStatus() (utils/api.ts)              │
├─────────┼────────────────────────────────────────────────────────────┤
│         ↓                                                            │
│                      EXPRESS API SERVER                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  server/index.ts + routes/jobs.ts                            │   │
│  │                                                               │   │
│  │  POST /api/jobs/upload-url  → Generate signed GCS URL        │   │
│  │  POST /api/jobs/start-job   → Publish Pub/Sub + Firestore    │   │
│  │  GET  /api/jobs/job-status/:id → Poll Firestore              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                    │                                       │
│         ↓                    ↓                                       │
├─────────┼────────────────────┼───────────────────────────────────────┤
│    GCS Bucket          Pub/Sub Topic                                 │
│    (Direct Upload)     (job-upload-topic)                            │
│         │                    │                                       │
│         │                    ↓                                       │
│         │           PYTHON WORKER (worker.py)                        │
│         │           ┌────────────────────────┐                       │
│         │           │  Pub/Sub Subscriber    │                       │
│         │           │  callback(message)     │                       │
│         │           └────────┬───────────────┘                       │
│         │                    │                                       │
│         │      ┌─────────────┴─────────────┐                         │
│         │      ↓                           ↓                         │
│         │  process_upload_local()   process_upload_vertex()          │
│         │  (Local ML Pipeline)      (Dispatch to Vertex AI)          │
│         │      ↓                           ↓                         │
│         └──────┤                    Vertex AI                        │
│                │                    CustomContainerTrainingJob       │
│                ↓                           ↓                         │
│           FIRESTORE                        │                         │
│           (jobs/{jobId})                   │                         │
│           - status: "uploading"            │                         │
│           - status: "processing"           │                         │
│           - status: "complete"             │                         │
│           - outliers: [...]                │                         │
│           - stats: {...}                   │                         │
│                ↑                           │                         │
│                └───────────────────────────┘                         │
│                                                                       │
│           Frontend Polls (2s interval)                               │
│                ↑                                                      │
└────────────────┴──────────────────────────────────────────────────────┘
```

### Current Data Flow

1. **Frontend → Signed URL Request:**
   - `uploadCsv()` → POST `/api/jobs/upload-url` with filename
   - Express generates UUID jobId, creates GCS signed URL (15min expiry)
   - Returns: `{url, jobId, gcsFileName}`

2. **Frontend → Direct GCS Upload:**
   - PUT to signed URL with file buffer
   - Browser uploads directly to GCS (bypasses Express)

3. **Frontend → Job Start:**
   - POST `/api/jobs/start-job` with `{jobId, gcsFileName}`
   - Express publishes Pub/Sub message: `{jobId, bucket, file}`
   - Express writes Firestore: `jobs/{jobId}` with `status: "uploading"`

4. **Worker → Processing:**
   - Pub/Sub subscriber receives message
   - Worker updates Firestore: `status: "processing"`
   - Processes locally OR dispatches to Vertex AI
   - Writes results to Firestore: `status: "complete"`, `outliers: [...]`, `stats: {...}`

5. **Frontend → Polling:**
   - GET `/api/jobs/job-status/:id` every 2 seconds
   - Reads from Firestore `jobs/{jobId}`
   - Stops when `status === "complete"` or `status === "error"`

## Integration Points for Production Features

### 1. Authentication Integration Points

**WHERE:** Express middleware layer (before all routes)

**Pattern:** API Key middleware (simpler than JWT for service-to-service, no session management needed)

**Implementation:**

```typescript
// server/middleware/auth.ts
import { Request, Response, NextFunction } from 'express';

export function apiKeyAuth(req: Request, res: Response, next: NextFunction) {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey) {
    return res.status(401).json({ error: 'Missing API key' });
  }

  // Compare with env var or validate against Firestore collection
  const validKeys = process.env.VALID_API_KEYS?.split(',') || [];

  if (!validKeys.includes(apiKey as string)) {
    return res.status(403).json({ error: 'Invalid API key' });
  }

  next();
}

// server/index.ts
app.use('/api', apiKeyAuth); // Protect all /api routes
```

**Integration Impact:**
- **NEW:** Add middleware BEFORE route definitions
- **MODIFY:** Frontend must send `X-API-Key` header in all API calls
- **NO CHANGE:** GCS signed URLs work independently (already authenticated by signature)

**Alternative Pattern (JWT):**
- Better for multi-user apps with login
- Requires session management + token refresh
- More complex: `jsonwebtoken` + `express-session`
- Overkill for demo/internal tool

**Build Order:** Implement authentication FIRST (foundation for rate limiting)

Sources:
- [API Authentication Done Right: JWTs, API Keys, and OAuth2 in Production (2026 Guide)](https://dev.to/young_gao/api-authentication-done-right-jwts-api-keys-and-oauth2-in-production-38a6)
- [API key vs JWT: Secure B2B SaaS with modern M2M authentication](https://www.scalekit.com/blog/apikey-jwt-comparison)

---

### 2. Rate Limiting Integration Points

**WHERE:** Express middleware layer (after authentication, before routes)

**Pattern:** `express-rate-limit` with per-endpoint limits

**Implementation:**

```typescript
// server/middleware/rateLimits.ts
import rateLimit from 'express-rate-limit';

// Upload endpoint: expensive operation, strict limit
export const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 uploads per 15 min per IP
  standardHeaders: 'draft-8',
  legacyHeaders: false,
  message: { error: 'Too many uploads. Please wait 15 minutes.' }
});

// Polling endpoint: frequent calls, higher limit
export const pollingLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute (1 per second)
  standardHeaders: 'draft-8',
  legacyHeaders: false,
  message: { error: 'Polling too frequently. Slow down.' }
});

// server/routes/jobs.ts
router.post('/upload-url', uploadLimiter, async (req, res) => { ... });
router.post('/start-job', uploadLimiter, async (req, res) => { ... });
router.get('/job-status/:id', pollingLimiter, async (req, res) => { ... });
```

**Integration Impact:**
- **NEW:** Add rate limit middleware per endpoint
- **MODIFY:** Frontend must handle 429 responses gracefully
- **CONSIDER:** IPv6 subnet bucketing (attackers can rotate IPs in /64 range)

**Build Order:** After authentication (rate limits can be per-user instead of per-IP)

Sources:
- [How to Add Rate Limiting to Express APIs](https://oneuptime.com/blog/post/2026-02-02-express-rate-limiting/view)
- [express-rate-limit - npm](https://www.npmjs.com/package/express-rate-limit)
- [How to Handle IPv6 in Rate Limiting Middleware](https://oneuptime.com/blog/post/2026-03-20-ipv6-in-rate-limiting-middleware/view)

---

### 3. CSV Input Validation Integration Points

**WHERE:** Two layers - Express (pre-upload) and Worker (pre-processing)

**Pattern:** Defense in depth - validate early, validate again before processing

**Express Layer (routes/jobs.ts):**

```typescript
import validator from 'validator';
import path from 'path';

router.post('/upload-url', async (req, res) => {
  const { filename, contentType } = req.body;

  // 1. Validate filename
  if (!filename || typeof filename !== 'string') {
    return res.status(400).json({ error: 'Invalid filename' });
  }

  // 2. Sanitize filename (prevent path traversal)
  const safeName = path.basename(filename); // Removes ../../../etc/passwd

  // 3. Validate file extension
  const ext = path.extname(safeName).toLowerCase();
  if (ext !== '.csv') {
    return res.status(400).json({ error: 'Only CSV files allowed' });
  }

  // 4. Validate content type
  const allowedTypes = ['text/csv', 'application/csv', 'text/plain'];
  if (!allowedTypes.includes(contentType)) {
    return res.status(400).json({ error: 'Invalid content type' });
  }

  // 5. Validate filename length
  if (safeName.length > 255) {
    return res.status(400).json({ error: 'Filename too long' });
  }

  // Continue with signed URL generation...
});
```

**Worker Layer (worker.py):**

```python
def process_upload_local(job_id, bucket_name, file_path):
    try:
        logger.info(f"Starting local processing for job {job_id}")

        # 1. Validate message fields
        if not job_id or not bucket_name or not file_path:
            raise ValueError("Missing required fields in Pub/Sub message")

        db.collection('jobs').document(job_id).set(
            {"status": "processing"}, merge=True
        )

        # 2. Download with size limit (prevent memory exhaustion)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Check size before download
        metadata = blob.reload()
        max_size = 50 * 1024 * 1024  # 50MB
        if blob.size > max_size:
            raise ValueError(f"File too large: {blob.size} bytes (max: {max_size})")

        csv_bytes = blob.download_as_bytes()

        # 3. Validate CSV structure
        try:
            df = loader.load_original_data(csv_bytes)
        except pd.errors.ParserError as e:
            raise ValueError(f"Invalid CSV format: {str(e)}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid file encoding: {str(e)}")

        # 4. Validate CSV content
        if df.empty:
            raise ValueError("CSV file is empty")

        if len(df.columns) == 0:
            raise ValueError("CSV has no columns")

        if len(df) > 100000:  # Arbitrary limit
            raise ValueError(f"Too many rows: {len(df)} (max: 100,000)")

        # Continue processing...

    except ValueError as e:
        # User-friendly validation errors
        logger.warning(f"Validation error for job {job_id}: {e}")
        db.collection('jobs').document(job_id).set({
            "status": "error",
            "error": str(e)
        }, merge=True)
    except Exception as e:
        # Unexpected errors
        logger.error(f"Error processing job {job_id}: {e}")
        db.collection('jobs').document(job_id).set({
            "status": "error",
            "error": "Internal processing error"
        }, merge=True)
```

**CSV Formula Injection Prevention:**

```python
# In worker.py after loading CSV
def sanitize_csv_cells(df):
    """Prevent CSV formula injection by escaping dangerous characters."""
    dangerous_chars = ['=', '+', '-', '@', '\t', '\r']

    for col in df.columns:
        if df[col].dtype == object:  # String columns
            df[col] = df[col].apply(lambda x:
                f"'{x}" if isinstance(x, str) and x and x[0] in dangerous_chars else x
            )
    return df

# Call before processing
df = sanitize_csv_cells(df)
```

**Integration Impact:**
- **NEW:** Add validation middleware in Express routes
- **NEW:** Add validation at start of worker processing
- **MODIFY:** Worker callback signature (add field validation)
- **NEW:** Firestore error schema (structured error messages)

**Build Order:** Foundation feature (blocks malicious/malformed uploads)

Sources:
- [Input Validation Security Best Practices for Node.js](https://www.nodejs-security.com/blog/input-validation-best-practices-for-nodejs)
- [Best-practice methods to prevent CSV formula injection attacks in Node.js](https://www.cyberchief.ai/2024/09/csv-formula-injection-attacks.html)
- [7 Best Practices for Sanitizing Input in Node.js](https://medium.com/devmap/7-best-practices-for-sanitizing-input-in-node-js-e61638440096)

---

### 4. Progress Tracking Integration Points

**WHERE:** Firestore schema + Worker state updates + Frontend polling logic

**Pattern:** Firestore real-time listeners (optional) + enriched status field

**Firestore Schema Changes:**

```typescript
// Current schema
interface JobDocument {
  status: "uploading" | "processing" | "complete" | "error";
  createdAt: Date;
  outliers?: any[];
  stats?: any;
  error?: string;
}

// NEW schema (backward compatible)
interface JobDocumentV2 {
  status: "uploading" | "processing" | "complete" | "error";
  createdAt: Date;
  updatedAt: Date;

  // NEW: Progress tracking
  progress?: {
    stage: "downloading" | "validating" | "preprocessing" | "training" | "scoring" | "saving";
    percent?: number; // 0-100
    message?: string;
  };

  // NEW: Metrics
  metrics?: {
    totalRows: number;
    keptColumns: number;
    droppedColumns: number;
    processingTimeMs?: number;
  };

  // Existing fields
  outliers?: any[];
  stats?: any;
  error?: string;
}
```

**Worker Updates (worker.py):**

```python
def update_progress(job_id, stage, percent=None, message=None):
    """Update job progress in Firestore."""
    progress_data = {
        "status": "processing",
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "progress": {
            "stage": stage,
        }
    }
    if percent is not None:
        progress_data["progress"]["percent"] = percent
    if message:
        progress_data["progress"]["message"] = message

    db.collection('jobs').document(job_id).set(progress_data, merge=True)

def process_upload_local(job_id, bucket_name, file_path):
    try:
        start_time = time.time()

        # Stage 1: Downloading
        update_progress(job_id, "downloading", 10, "Downloading CSV from storage")
        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()

        # Stage 2: Validating
        update_progress(job_id, "validating", 20, "Validating CSV format")
        df = loader.load_original_data(csv_bytes)

        # Stage 3: Preprocessing
        update_progress(job_id, "preprocessing", 30, "Applying Rule of 9 filter")
        process_df = df.fillna("missing").astype(str)
        # ... Rule of 9 logic ...

        # Stage 4: Training
        update_progress(job_id, "training", 50, "Training autoencoder model")
        keras_model.fit(X_train, X_train, epochs=15, ...)

        # Stage 5: Scoring
        update_progress(job_id, "scoring", 80, "Computing outlier scores")
        reconstruction = keras_model.predict(vectorized_df)
        mse = np.mean(np.power(vectorized_df - reconstruction, 2), axis=1)

        # Stage 6: Saving
        update_progress(job_id, "saving", 95, "Saving results")
        processing_time = int((time.time() - start_time) * 1000)

        db.collection('jobs').document(job_id).set({
            "status": "complete",
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "stats": stats,
            "outliers": outliers_data,
            "metrics": {
                "totalRows": len(df),
                "keptColumns": len(cols_to_keep),
                "droppedColumns": len(stats['ignored_columns']),
                "processingTimeMs": processing_time
            },
            "progress": {
                "stage": "complete",
                "percent": 100,
                "message": "Analysis complete"
            }
        }, merge=True)
```

**Frontend Changes (Index.tsx):**

```typescript
// Option 1: Continue polling (simple, works with current code)
useEffect(() => {
  if (!jobId || status === "complete" || status === "error") return;

  const interval = setInterval(async () => {
    const data = await checkJobStatus(jobId);

    // NEW: Update progress state
    if (data.progress) {
      setProgressStage(data.progress.stage);
      setProgressPercent(data.progress.percent || 0);
      setProgressMessage(data.progress.message);
    }

    if (data.status === "complete") {
      setResults(data.outliers || []);
      setStats(data.stats);
      setStatus("complete");
      clearInterval(interval);
    }
  }, 2000);

  return () => clearInterval(interval);
}, [jobId, status]); // Remove `toast` from dependencies

// Option 2: Real-time listeners (better UX, more complex)
import { doc, onSnapshot } from 'firebase/firestore';

useEffect(() => {
  if (!jobId) return;

  const unsubscribe = onSnapshot(doc(firestore, 'jobs', jobId), (doc) => {
    const data = doc.data();
    if (data?.progress) {
      setProgressStage(data.progress.stage);
      setProgressPercent(data.progress.percent || 0);
    }
    if (data?.status === 'complete') {
      setResults(data.outliers);
      setStatus('complete');
    }
  });

  return () => unsubscribe();
}, [jobId]);
```

**UI Component (new):**

```typescript
// components/ProgressIndicator.tsx
export function ProgressIndicator({ stage, percent, message }: {
  stage: string;
  percent?: number;
  message?: string;
}) {
  const stageLabels = {
    downloading: "Downloading CSV",
    validating: "Validating format",
    preprocessing: "Preparing data",
    training: "Training model",
    scoring: "Computing scores",
    saving: "Saving results"
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>{stageLabels[stage] || stage}</span>
        {percent && <span>{percent}%</span>}
      </div>
      <Progress value={percent || 0} />
      {message && <p className="text-xs text-muted-foreground">{message}</p>}
    </div>
  );
}
```

**Integration Impact:**
- **MODIFY:** Firestore schema (backward compatible - old jobs still work)
- **MODIFY:** Worker adds progress updates throughout processing
- **MODIFY:** Frontend displays progress bar instead of generic spinner
- **NEW:** ProgressIndicator component
- **CONSIDER:** Real-time listeners (requires Firestore client SDK in frontend)

**Build Order:** Mid-priority polish feature (after validation, before CSV export)

Sources:
- [How to Set Up Real-Time Listeners for Live Data Updates in Firestore](https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-real-time-listeners-for-live-data-updates-in-firestore/view)
- [Get realtime updates with Cloud Firestore](https://firebase.google.com/docs/firestore/query-data/listen)

---

### 5. Job Cancellation Integration Points

**WHERE:** Frontend UI button → Express endpoint → Firestore flag → Worker check + Vertex AI cancellation

**Pattern:** Cooperative cancellation via Firestore flag

**Firestore Schema:**

```typescript
interface JobDocumentV2 {
  // ... existing fields ...
  cancelRequested?: boolean;
  cancelledAt?: Date;
}
```

**Express Endpoint (routes/jobs.ts):**

```typescript
router.post('/cancel-job', async (req, res) => {
  try {
    const { jobId } = req.body;

    // Validate jobId
    if (!jobId || typeof jobId !== 'string') {
      return res.status(400).json({ error: 'Invalid jobId' });
    }

    // Check if job exists
    const jobDoc = await firestore.collection('jobs').doc(jobId).get();
    if (!jobDoc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const jobData = jobDoc.data();

    // Can't cancel completed/errored jobs
    if (jobData?.status === 'complete' || jobData?.status === 'error') {
      return res.status(400).json({ error: 'Job already finished' });
    }

    // Set cancellation flag
    await firestore.collection('jobs').doc(jobId).set({
      cancelRequested: true,
      cancelledAt: new Date(),
      status: 'error',
      error: 'Cancelled by user'
    }, { merge: true });

    // TODO: Cancel Vertex AI job if in vertex mode
    // This requires storing the Vertex AI job name in Firestore

    res.json({ success: true, message: 'Job cancelled' });
  } catch (error) {
    console.error('Error cancelling job:', error);
    res.status(500).json({ error: 'Failed to cancel job' });
  }
});
```

**Worker Changes (worker.py):**

```python
def check_cancellation(job_id):
    """Check if job has been cancelled."""
    doc = db.collection('jobs').document(job_id).get()
    if doc.exists:
        data = doc.to_dict()
        if data.get('cancelRequested'):
            raise CancellationError(f"Job {job_id} cancelled by user")

class CancellationError(Exception):
    pass

def process_upload_local(job_id, bucket_name, file_path):
    try:
        # Check cancellation at multiple points
        check_cancellation(job_id)

        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()

        check_cancellation(job_id)

        df = loader.load_original_data(csv_bytes)
        # ... preprocessing ...

        check_cancellation(job_id)

        # Training (can't check during fit, but check before)
        keras_model.fit(X_train, X_train, epochs=15, ...)

        check_cancellation(job_id)

        # ... scoring and saving ...

    except CancellationError as e:
        logger.info(f"Job {job_id} cancelled")
        # Firestore already updated by cancellation endpoint
        return
    except Exception as e:
        # ... normal error handling ...
```

**Vertex AI Cancellation (process_upload_vertex):**

```python
def process_upload_vertex(job_id, bucket_name, file_path):
    from google.cloud import aiplatform

    try:
        logger.info(f"Starting Vertex AI job {job_id}")

        # ... existing setup ...

        job = aiplatform.CustomContainerTrainingJob(...)
        vertex_job = job.run(...)  # This is async

        # Store Vertex AI job name for cancellation
        db.collection('jobs').document(job_id).set({
            "vertexJobName": vertex_job.resource_name
        }, merge=True)

        logger.info(f"Job submitted to Vertex AI: {vertex_job.resource_name}")

    except Exception as e:
        # ... error handling ...

# NEW: Cancellation helper
def cancel_vertex_job(vertex_job_name):
    """Cancel a running Vertex AI job."""
    from google.cloud import aiplatform

    try:
        aiplatform.init(project=PROJECT_ID, location="us-central1")
        job = aiplatform.CustomJob(vertex_job_name)
        job.cancel()
        logger.info(f"Cancelled Vertex AI job: {vertex_job_name}")
    except Exception as e:
        logger.error(f"Failed to cancel Vertex AI job: {e}")
```

**Express Cancellation (updated):**

```typescript
router.post('/cancel-job', async (req, res) => {
  // ... validation ...

  const jobData = jobDoc.data();

  // If job is running on Vertex AI, cancel it
  if (jobData?.vertexJobName) {
    // Call Python worker endpoint or Cloud Function to cancel
    // OR implement Vertex AI cancellation in Node.js
    const aiplatform = require('@google-cloud/aiplatform');
    const client = new aiplatform.JobServiceClient();

    try {
      await client.cancelCustomJob({ name: jobData.vertexJobName });
    } catch (error) {
      console.error('Failed to cancel Vertex AI job:', error);
      // Continue anyway - set Firestore flag
    }
  }

  // Set cancellation flag...
});
```

**Frontend UI (Index.tsx):**

```typescript
const handleCancel = useCallback(async () => {
  if (!jobId) return;

  try {
    await fetch(`${API_BASE}/api/jobs/cancel-job`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ jobId })
    });

    setStatus('error');
    setError('Job cancelled');
    toast({ title: 'Job Cancelled', variant: 'destructive' });
  } catch (err) {
    console.error('Failed to cancel job:', err);
  }
}, [jobId]);

// In render:
{isProcessing && (
  <div>
    <ProgressIndicator ... />
    <Button onClick={handleCancel} variant="destructive">Cancel Job</Button>
  </div>
)}
```

**Integration Impact:**
- **NEW:** Cancel endpoint in Express
- **NEW:** Cancellation checks in worker
- **MODIFY:** Firestore schema (add cancelRequested, vertexJobName)
- **NEW:** Cancel button in frontend
- **CONSIDER:** Vertex AI job name storage (for cloud mode)

**Build Order:** Polish feature (after progress tracking, before CSV export)

---

### 6. CSV Export Integration Points

**WHERE:** Frontend download button → Express endpoint → Firestore data → CSV generation

**Pattern:** Generate CSV from Firestore results (don't regenerate from GCS)

**Express Endpoint (routes/jobs.ts):**

```typescript
import { stringify } from 'csv-stringify/sync';

router.get('/export-results/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Get job from Firestore
    const jobDoc = await firestore.collection('jobs').doc(id).get();

    if (!jobDoc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const jobData = jobDoc.data();

    if (jobData?.status !== 'complete') {
      return res.status(400).json({ error: 'Job not complete' });
    }

    if (!jobData?.outliers || jobData.outliers.length === 0) {
      return res.status(404).json({ error: 'No results to export' });
    }

    // Convert outliers array to CSV
    const csv = stringify(jobData.outliers, {
      header: true,
      columns: ['reconstruction_error', ...Object.keys(jobData.outliers[0]).filter(k => k !== 'reconstruction_error')]
    });

    // Send as downloadable file
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="outliers_${id}.csv"`);
    res.send(csv);

  } catch (error) {
    console.error('Error exporting results:', error);
    res.status(500).json({ error: 'Failed to export results' });
  }
});
```

**Frontend (Index.tsx):**

```typescript
const handleExport = useCallback(async () => {
  if (!jobId) return;

  try {
    const response = await fetch(`${API_BASE}/api/jobs/export-results/${jobId}`);

    if (!response.ok) throw new Error('Export failed');

    // Create download link
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `outliers_${jobId}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    toast({ title: 'Export Successful' });
  } catch (err) {
    console.error('Export error:', err);
    toast({ title: 'Export Failed', variant: 'destructive' });
  }
}, [jobId]);

// In results UI:
<Button onClick={handleExport} variant="outline">
  Download Results CSV
</Button>
```

**Alternative Pattern (Client-Side Export):**

```typescript
// No server endpoint needed - generate CSV in browser
import { unparse } from 'papaparse';

const handleExportClientSide = useCallback(() => {
  if (!results || results.length === 0) return;

  const csv = unparse(results, {
    columns: ['reconstruction_error', ...Object.keys(results[0]).filter(k => k !== 'reconstruction_error')]
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `outliers_${jobId}.csv`;
  a.click();
  window.URL.revokeObjectURL(url);
}, [results, jobId]);
```

**Integration Impact:**
- **OPTION 1 (Server-side):** NEW export endpoint, frontend calls API
- **OPTION 2 (Client-side):** NEW frontend logic, no backend changes
- **RECOMMEND:** Client-side (simpler, no server load, data already in frontend)

**Build Order:** Polish feature (independent, can be done anytime after results display)

---

### 7. Error Boundaries Integration Points

**WHERE:** React component tree (wrap App and/or Index)

**Pattern:** react-error-boundary library for functional components

**Implementation:**

```typescript
// App.tsx (MODIFY)
import { ErrorBoundary } from 'react-error-boundary';
import { Toaster } from "@/components/ui/toaster";

function ErrorFallback({ error, resetErrorBoundary }: {
  error: Error;
  resetErrorBoundary: () => void;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="max-w-md bg-white rounded-2xl shadow-lg p-8">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h2>
        <p className="text-gray-700 mb-4">
          The application encountered an unexpected error. Please try again.
        </p>
        <details className="mb-4">
          <summary className="cursor-pointer text-sm text-gray-500">Error details</summary>
          <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-auto">
            {error.message}
          </pre>
        </details>
        <Button onClick={resetErrorBoundary}>Reload Application</Button>
      </div>
    </div>
  );
}

const App = () => (
  <ErrorBoundary
    FallbackComponent={ErrorFallback}
    onReset={() => window.location.href = '/'}
    onError={(error, errorInfo) => {
      // Log to error reporting service (e.g., Sentry)
      console.error('Error boundary caught:', error, errorInfo);
    }}
  >
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <Index />
      </TooltipProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);
```

**Granular Boundaries (optional):**

```typescript
// Index.tsx - wrap high-risk components
<ErrorBoundary
  FallbackComponent={() => <div>Failed to load results</div>}
  resetKeys={[jobId]} // Reset when jobId changes
>
  <PreviewTable rows={results} ... />
</ErrorBoundary>
```

**Async Error Handling (CSV parsing):**

```typescript
// utils/csv-parser.ts - throw errors to error boundary
export async function parseCSVFile(file: File): Promise<CSVParseResult> {
  try {
    const text = await file.text();
    const parsed = Papa.parse(text, { ... });

    if (parsed.errors.length > 0) {
      throw new Error(`CSV parse errors: ${parsed.errors[0].message}`);
    }

    return { rows: parsed.data, ... };
  } catch (error) {
    // Re-throw to be caught by error boundary
    throw new Error(`Failed to parse CSV: ${error.message}`);
  }
}
```

**Integration Impact:**
- **NEW:** Add `react-error-boundary` dependency
- **MODIFY:** Wrap App in ErrorBoundary
- **NEW:** ErrorFallback component
- **CONSIDER:** Granular boundaries for Preview/Results components
- **FIX:** Remove `toast` from polling useEffect dependencies (causes interval churn)

**Build Order:** Foundation feature (prevents white screen of death)

Sources:
- [Error Boundaries in React: The Safety Net Every Production App Needs](https://saraswathi-mac.medium.com/error-boundaries-in-react-the-safety-net-every-production-app-needs-f85809bd5563)
- [How to Implement React Error Boundaries for Resilient UIs](https://oneuptime.com/blog/post/2026-02-20-react-error-boundaries/view)

---

## Component Changes Summary

### NEW Components/Files

| Component | Purpose | Dependencies |
|-----------|---------|--------------|
| `server/middleware/auth.ts` | API key authentication | None |
| `server/middleware/rateLimits.ts` | Rate limiting configs | `express-rate-limit` |
| `client/components/ProgressIndicator.tsx` | Progress bar UI | `shadcn/ui Progress` |
| `client/components/ErrorFallback.tsx` | Error boundary UI | `react-error-boundary` |

### MODIFIED Components

| Component | Changes | Reason |
|-----------|---------|--------|
| `server/index.ts` | Add auth/rate limit middleware | Security hardening |
| `server/routes/jobs.ts` | Add validation, export, cancel endpoints | Input validation + new features |
| `worker.py` | Add validation, progress updates, cancellation checks | Robustness + UX |
| `client/App.tsx` | Wrap in ErrorBoundary | Prevent crashes |
| `client/pages/Index.tsx` | Add progress, cancel, export UI | Feature completeness |
| `client/utils/api.ts` | Add cancel/export functions | New endpoints |

### Firestore Schema Evolution

```typescript
// v1 (current)
{
  status: string,
  createdAt: Date,
  outliers?: any[],
  stats?: any,
  error?: string
}

// v2 (production-ready)
{
  status: string,
  createdAt: Date,
  updatedAt: Date,

  // Progress tracking
  progress?: {
    stage: string,
    percent: number,
    message: string
  },

  // Metrics
  metrics?: {
    totalRows: number,
    keptColumns: number,
    droppedColumns: number,
    processingTimeMs: number
  },

  // Cancellation
  cancelRequested?: boolean,
  cancelledAt?: Date,
  vertexJobName?: string,

  // Results
  outliers?: any[],
  stats?: any,
  error?: string
}
```

## Build Order Recommendation

**Phase 1: Security Foundation (MUST DO FIRST)**
1. Input validation (Express + Worker)
2. API authentication middleware
3. Error boundaries (prevent crashes)
4. CORS restrictions (lock down origins)

**Phase 2: Reliability (BEFORE PUBLIC RELEASE)**
5. Rate limiting (prevent abuse)
6. Worker message validation (fail fast)
7. Environment variable checks (startup validation)

**Phase 3: UX Polish (AFTER SECURITY)**
8. Progress tracking (Firestore + Worker + Frontend)
9. Job cancellation (Firestore flag + Worker checks)
10. CSV export (client-side, simple)

**Phase 4: Operational Improvements (ONGOING)**
11. Monitoring/logging improvements
12. Performance optimization
13. Real-time listeners (Firestore, optional upgrade)

## Data Flow Changes

**Before (v0.1):**
```
Frontend → Signed URL → GCS → Pub/Sub → Worker → Firestore
         ↑                                            ↓
         └────────────── Poll (2s) ──────────────────┘
```

**After (v1.0 Production):**
```
Frontend → [Auth] → [Rate Limit] → [Validate] → Signed URL → GCS
         ↑                                                     ↓
         │                                           Pub/Sub (validated message)
         │                                                     ↓
         │                                           Worker [Validate CSV]
         │                                                     ↓
         │                                           [Check Cancellation]
         │                                                     ↓
         │                                           [Update Progress]
         │                                                     ↓
         │                                           Firestore (enriched schema)
         │                                                     ↓
         └─────── Poll OR Real-time Listener ─────────────────┘
                        ↓
                  [Export CSV] [Cancel Job]
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Validating Only in Frontend

**What people do:** Check file type, size in browser but not on server
**Why it's wrong:** Client-side validation can be bypassed with curl/Postman
**Do this instead:** Validate in both places - frontend for UX, backend for security

### Anti-Pattern 2: Storing API Keys in Firestore

**What people do:** Store valid API keys in Firestore for "flexible management"
**Why it's wrong:** Firestore reads are billed, adds latency, creates attack surface
**Do this instead:** Use environment variables or Secret Manager, validate in middleware

### Anti-Pattern 3: Re-reading GCS for Export

**What people do:** Download original CSV from GCS, re-score, export
**Why it's wrong:** Unnecessary compute + storage reads, slow, expensive
**Do this instead:** Export directly from Firestore results (already computed)

### Anti-Pattern 4: Polling After Completion

**What people do:** Continue polling even after job is complete
**Why it's wrong:** Wastes API calls, hits rate limits, burns Firestore reads
**Do this instead:** Stop polling when status is "complete" or "error"

### Anti-Pattern 5: Acking Pub/Sub Before Processing

**What people do:** Ack message immediately, then process (prevents redelivery)
**Why it's wrong:** If worker crashes mid-processing, job is lost forever
**Do this instead:** Extend ack deadline during processing, ack only after Firestore write

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-100 users/day | Current architecture sufficient. Single worker instance. |
| 100-1K users/day | Add worker autoscaling (Cloud Run or GKE). Add Firestore index on `status` + `createdAt`. |
| 1K-10K users/day | Move to Cloud Run for worker (auto-scales). Add Redis for rate limiting (shared state). Consider job queue priority. |
| 10K+ users/day | Add CDN for frontend. Separate Pub/Sub topics by priority. Consider BigQuery for analytics. Add monitoring/alerting. |

**First bottleneck:** Worker processing (single instance can't keep up)
**Solution:** Deploy worker on Cloud Run with min/max instances, or GKE with HPA

**Second bottleneck:** Firestore polling (too many reads)
**Solution:** Switch to real-time listeners or WebSockets for status updates

---

## Sources

Authentication:
- [API Authentication Done Right: JWTs, API Keys, and OAuth2 in Production (2026 Guide)](https://dev.to/young_gao/api-authentication-done-right-jwts-api-keys-and-oauth2-in-production-38a6)
- [API key vs JWT: Secure B2B SaaS with modern M2M authentication](https://www.scalekit.com/blog/apikey-jwt-comparison)

Rate Limiting:
- [How to Add Rate Limiting to Express APIs](https://oneuptime.com/blog/post/2026-02-02-express-rate-limiting/view)
- [express-rate-limit - npm](https://www.npmjs.com/package/express-rate-limit)
- [How to Handle IPv6 in Rate Limiting Middleware](https://oneuptime.com/blog/post/2026-03-20-ipv6-in-rate-limiting-middleware/view)

Error Boundaries:
- [Error Boundaries in React: The Safety Net Every Production App Needs](https://saraswathi-mac.medium.com/error-boundaries-in-react-the-safety-net-every-production-app-needs-f85809bd5563)
- [How to Implement React Error Boundaries for Resilient UIs](https://oneuptime.com/blog/post/2026-02-20-react-error-boundaries/view)

Progress Tracking:
- [How to Set Up Real-Time Listeners for Live Data Updates in Firestore](https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-real-time-listeners-for-live-data-updates-in-firestore/view)
- [Get realtime updates with Cloud Firestore](https://firebase.google.com/docs/firestore/query-data/listen)

Input Validation:
- [Input Validation Security Best Practices for Node.js](https://www.nodejs-security.com/blog/input-validation-best-practices-for-nodejs)
- [Best-practice methods to prevent CSV formula injection attacks in Node.js](https://www.cyberchief.ai/2024/09/csv-formula-injection-attacks.html)
- [7 Best Practices for Sanitizing Input in Node.js](https://medium.com/devmap/7-best-practices-for-sanitizing-input-in-node-js-e61638440096)

---

*Architecture research for: AutoEncoder Production Features Integration*
*Researched: 2026-03-24*
