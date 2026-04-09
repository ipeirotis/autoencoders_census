import { useQuery } from '@tanstack/react-query';

// Match the API_BASE convention used by frontend/client/utils/api.ts so
// that split frontend/backend deployments (where the API runs on a
// different origin) reach the API server instead of the static frontend.
// Defaults to '' (same origin) for the standard `npm run dev:server` setup.
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

/** HTTP status codes that indicate the request will never succeed for this
 *  job/session, so continuing to poll is pointless and wasteful. */
const NON_RETRIABLE_STATUSES = new Set([401, 403, 404]);

/** Custom error that preserves the HTTP status code from the server response
 *  so callers (e.g. refetchInterval) can distinguish retriable from terminal. */
class HttpError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = 'HttpError';
  }
}

interface JobStatus {
  jobId: string;
  // Must mirror worker.py's JobStatus enum - see StageIndicator for details.
  status: 'queued' | 'processing' | 'training' | 'scoring' | 'complete' | 'error' | 'canceled';
  stageProgress?: number;  // 0-100 percent for current stage
  overallProgress?: number;  // 0-100 percent for entire job
  fileName?: string;
  fileSize?: number;
  createdAt: string;
  updatedAt: string;
  error?: string;
  errorType?: string;
}

/**
 * Hook for polling job status with TanStack Query.
 *
 * Features:
 * - Automatic polling every 2 seconds for active jobs
 * - Stops polling when job reaches terminal state (complete/error/canceled)
 * - Stops polling on non-retriable HTTP errors (401/403/404)
 * - Keeps polling through transient errors (5xx, network blips)
 * - Automatic cleanup on component unmount (no memory leaks)
 * - Conditional fetching (only polls if jobId exists)
 *
 * @param jobId - The job ID to poll, or null to disable polling
 * @returns TanStack Query result with job status data
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
    enabled: !!jobId, // Don't poll if no jobId (FE-10)
    refetchInterval: (query) => {
      // Stop polling on non-retriable HTTP errors (session expired, job
      // not found/forbidden). Transient errors (5xx, network blips) keep
      // polling so the page recovers automatically after a brief outage.
      const err = query.state.error;
      if (err instanceof HttpError && NON_RETRIABLE_STATUSES.has(err.status)) {
        return false;
      }

      const status = query.state.data?.status;

      // Terminal states - stop polling (FE-09)
      if (status === 'complete' || status === 'error' || status === 'canceled') {
        return false;
      }

      // Queued/processing - poll every 2 seconds
      return 2000;
    },
    // TanStack Query automatically stops on unmount (FE-08)
  });
}
