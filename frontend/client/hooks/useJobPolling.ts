import { useQuery } from '@tanstack/react-query';

// Match the API_BASE convention used by frontend/client/utils/api.ts so
// that split frontend/backend deployments (where the API runs on a
// different origin) reach the API server instead of the static frontend.
// Defaults to '' (same origin) for the standard `npm run dev:server` setup.
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

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
        throw new Error('Failed to fetch job status');
      }

      return response.json();
    },
    enabled: !!jobId, // Don't poll if no jobId (FE-10)
    refetchInterval: (query) => {
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
