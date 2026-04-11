import { useQuery } from '@tanstack/react-query';

// Match the API_BASE convention used by frontend/client/utils/api.ts so
// that split frontend/backend deployments reach the correct origin.
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

interface JobStatus {
  jobId: string;
  status: 'queued' | 'processing' | 'training' | 'scoring' | 'complete' | 'error' | 'canceled';
  stageProgress?: number;  // 0-100 percent for current stage
  overallProgress?: number;  // 0-100 percent for entire job
  fileName?: string;
  fileSize?: number;
  createdAt: string;
  updatedAt: string;
  // TASKS.md 2.3: structured error fields written by worker.mark_job_error.
  // `error` is the human-readable message, `errorCode` is a stable
  // machine-readable identifier (e.g. "csv_too_large", "no_usable_columns"),
  // and `errorType` buckets the code into a pipeline stage
  // ("validation" | "processing" | "training" | "scoring" | "internal").
  error?: string;
  errorCode?: string;
  errorType?: string;
}

/** Custom error that preserves the HTTP status code. */
class HttpError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = 'HttpError';
  }
}

/** HTTP status codes where continuing to poll is pointless. */
const NON_RETRIABLE_STATUSES = new Set([400, 401, 403, 404]);

/**
 * Hook for polling job status with TanStack Query.
 */
export function useJobPolling(jobId: string | null) {
  return useQuery<JobStatus>({
    queryKey: ['jobStatus', jobId],
    queryFn: async () => {
      if (!jobId) {
        throw new Error('Job ID is required');
      }

      const response = await fetch(`${API_BASE}/api/jobs/job-status/${jobId}`, {
        credentials: 'include',
      });

      if (!response.ok) {
        throw new HttpError('Failed to fetch job status', response.status);
      }

      return response.json();
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      // Stop polling on non-retriable HTTP errors (401 = session expired,
      // 404 = job not found). Transient 5xx errors keep polling.
      const err = query.state.error;
      if (err instanceof HttpError && NON_RETRIABLE_STATUSES.has(err.status)) {
        return false;
      }

      const status = query.state.data?.status;

      // Terminal states - stop polling
      if (status === 'complete' || status === 'error' || status === 'canceled') {
        return false;
      }

      // Queued/processing - poll every 2 seconds
      return 2000;
    },
    // TanStack Query automatically stops on unmount (FE-08)
  });
}
