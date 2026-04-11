/**
 * Frontend API client - calls Express endpoints
 *
 * This orchestrates the 3-step upload:
 * 1. getUploadUrl()  → POST /api/jobs/upload-url  → Gets signed GCS URL
 * 2. uploadToGcs()   → PUT to signed URL          → Uploads file directly to GCS
 * 3. startJob()      → POST /api/jobs/start-job   → Triggers Pub/Sub message
 *
 * All credentialed requests include the session cookie via
 * `credentials: 'include'` so they work both same-origin and on cross-site
 * deployments where the SPA and API live on different hosts.
 */
export interface DemoResponse {
  message: string;
}

export interface UploadResponse {
  jobId: string;
}

export interface JobStatus {
  // "uploading" is a client-only idle state used by Index.tsx. The server
  // reports the JobStatus enum values from worker.py (queued, processing,
  // training, scoring, complete, error, canceled).
  status:
    | "uploading"
    | "queued"
    | "processing"
    | "training"
    | "scoring"
    | "complete"
    | "error"
    | "canceled";
  stats?: any;
  outliers?: any[];
  // TASKS.md 2.3: structured error fields written by worker.mark_job_error.
  // `error` is the human-readable message; `errorCode` is a stable
  // machine-readable identifier (see utils/jobErrors.ts); `errorType`
  // buckets the code into a pipeline stage.
  error?: string;
  errorCode?: string;
  errorType?: string;
}

/**
 * Thrown when a job API request fails. Carries the HTTP status so callers
 * can distinguish authentication failures (401, session expired - the user
 * needs to log back in) from validation errors and from real server errors.
 */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function parseError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (body?.details?.[0]?.message) return body.details[0].message;
    if (typeof body?.error === "string") return body.error;
  } catch {
    // non-JSON error body, fall through
  }
  return fallback;
}

// 1. Get the Signed URL
async function getUploadUrl(filename: string, contentType: string) {
  const res = await fetch(`${API_BASE}/api/jobs/upload-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ filename, contentType }),
  });
  if (!res.ok) {
    throw new ApiError(await parseError(res, "Failed to get upload URL"), res.status);
  }
  return res.json();
}

// 2. Upload File to GCS
// The Content-Type used for the PUT MUST match whatever the server used
// when signing the URL - otherwise GCS rejects the upload. The server
// echoes nothing back; instead it signs using exactly the contentType
// we sent in step 1 (or omits the constraint entirely if we sent an
// empty one). We pass the same `file.type` through here for both calls,
// so the PUT header always matches the signing value.
async function uploadToGcs(url: string, file: File) {
  const res = await fetch(url, {
    method: "PUT",
    body: file,
    headers: { "Content-Type": file.type },
  });
  if (!res.ok) throw new Error("Failed to upload file to storage");
}

// 3. Notify Backend to Start Worker
async function startJob(jobId: string, gcsFileName: string) {
  const res = await fetch(`${API_BASE}/api/jobs/start-job`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ jobId, gcsFileName }),
  });
  if (!res.ok) {
    throw new ApiError(await parseError(res, "Failed to start processing job"), res.status);
  }
}

// Main Upload Function
export async function uploadCsv(file: File): Promise<UploadResponse> {
  // A. Get URL - pass file.type so the server signs for that exact
  //    Content-Type (or omits the constraint if file.type is empty).
  const { url, jobId, gcsFileName } = await getUploadUrl(file.name, file.type);

  // B. Upload (sends the same file.type header)
  await uploadToGcs(url, file);

  // C. Start Job
  await startJob(jobId, gcsFileName);

  return { jobId };
}

// Poll Status
export async function checkJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/job-status/${jobId}`, {
    credentials: "include",
  });
  if (!res.ok) {
    throw new ApiError(await parseError(res, "Failed to check job status"), res.status);
  }
  return res.json();
}
