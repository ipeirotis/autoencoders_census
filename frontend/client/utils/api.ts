export interface DemoResponse {
  message: string;
}

export interface UploadResponse {
  jobId: string;
}

export interface JobStatus {
  status: "uploading" | "processing" | "complete" | "error";
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
    body: JSON.stringify({ filename, contentType }),
  });
  if (!res.ok) throw new Error("Failed to get upload URL");
  return res.json();
}

// 2. Upload File to GCS
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
    body: JSON.stringify({ jobId, gcsFileName }),
  });
  if (!res.ok) throw new Error("Failed to start processing job");
}

// Main Upload Function
export async function uploadCsv(file: File): Promise<UploadResponse> {
  // A. Get URL
  const { url, jobId, gcsFileName } = await getUploadUrl(file.name, file.type);
  
  // B. Upload
  await uploadToGcs(url, file);
  
  // C. Start Job
  await startJob(jobId, gcsFileName);
  
  return { jobId };
}

// Poll Status
export async function checkJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/job-status/${jobId}`);
  if (!res.ok) throw new Error("Failed to check job status");
  return res.json();
}