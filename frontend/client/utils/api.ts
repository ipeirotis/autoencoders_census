/**
 * Frontend API client - calls Express endpoints
 *
 * This orchestrates the 3-step upload:
 * 1. getUploadUrl()  → POST /api/jobs/upload-url  → Gets signed GCS URL
 * 2. uploadToGcs()   → PUT to signed URL          → Uploads file directly to GCS
 * 3. startJob()      → POST /api/jobs/start-job   → Triggers Pub/Sub message
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
  error?: string;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

// 1. Get the Signed URL
async function getUploadUrl(filename: string, contentType: string) {
  const res = await fetch(`${API_BASE}/api/jobs/upload-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ filename, contentType }),
  });
  if (!res.ok) throw new Error("Failed to get upload URL");
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
  if (!res.ok) throw new Error("Failed to start processing job");
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
  if (!res.ok) throw new Error("Failed to check job status");
  return res.json();
}
