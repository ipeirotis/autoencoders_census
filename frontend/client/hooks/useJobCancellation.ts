import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useToast } from "@/hooks/use-toast";

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Hook for job cancellation with TanStack Query mutation.
 */
export function useJobCancellation() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (jobId: string) => {
      const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
        method: 'DELETE',
        credentials: 'include',
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
