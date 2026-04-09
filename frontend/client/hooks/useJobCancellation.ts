import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useToast } from "@/hooks/use-toast";

// Match the API_BASE convention used by frontend/client/utils/api.ts so
// that split frontend/backend deployments reach the API server.
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Hook for job cancellation with TanStack Query mutation.
 *
 * Features:
 * - Sends DELETE request to /api/jobs/:id
 * - Invalidates job status query cache on success (triggers refetch)
 * - Shows toast notifications for success/error
 *
 * Implements FE-06, FE-07 requirements (job cancellation capability).
 */
export function useJobCancellation() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (jobId: string) => {
      const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to cancel job');
      return response.json();
    },
    onSuccess: (data, jobId) => {
      // Invalidate job status query to trigger refetch
      queryClient.invalidateQueries({ queryKey: ['jobStatus', jobId] });
      toast({
        title: "Job canceled",
        description: "The job has been canceled successfully."
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: "Failed to cancel job. Please try again.",
        variant: "destructive"
      });
    }
  });
}
