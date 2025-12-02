export interface DemoResponse {
  message: string;
}

export interface UploadResponse {
  dataset_id: string;
  schema?: Array<{ name: string; detected_type: string }>;
  preview?: Record<string, unknown>[];
}

// Import from shared
//import { UploadResponse } from "@shared/api";  


// Uploads CSV file to backend and returns response
export async function uploadCsv(
  file: File,
  skipRows: number = 0
): Promise<UploadResponse> {
  const base =
    import.meta.env.VITE_API_BASE_URL ||
    "";

  const form = new FormData();
  form.append("file", file);
  form.append("skip_rows", String(skipRows));

  const url = `${base}/api/upload`;
  console.log("Uploading to URL:", url);
  console.log("Base URL:", base);

  const res = await fetch(url, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || "Upload failed");
  }

  return res.json();
}
