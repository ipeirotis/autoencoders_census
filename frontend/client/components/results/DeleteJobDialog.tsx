import { useState } from "react";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";

interface DeleteJobDialogProps {
  jobId: string;
}

export function DeleteJobDialog({ jobId }: DeleteJobDialogProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const handleDelete = async () => {
    setIsDeleting(true);

    try {
      const response = await fetch(`/api/jobs/${jobId}/files`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete files');
      }

      const data = await response.json();

      // Invalidate job query to trigger refetch (updates filesExpired flag)
      queryClient.invalidateQueries({ queryKey: ['jobStatus', jobId] });

      toast({
        title: "Files deleted",
        description: `Deleted ${data.filesDeleted} file(s) successfully.`
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete files. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Trash2 className="h-4 w-4 mr-2" />
          Delete Files
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete job files?</AlertDialogTitle>
          <AlertDialogDescription>
            This will permanently delete the uploaded CSV and result files from cloud storage.
            Job metadata will be preserved for history. This action cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={handleDelete} disabled={isDeleting}>
            {isDeleting ? 'Deleting...' : 'Delete Files'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
