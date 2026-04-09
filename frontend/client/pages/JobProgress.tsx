import { useParams } from 'react-router-dom';
import { useJobPolling } from '@/hooks/useJobPolling';
import { useJobCancellation } from '@/hooks/useJobCancellation';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

import { StageIndicator } from '@/components/progress/StageIndicator';
import { DualProgressBar } from '@/components/progress/DualProgressBar';
import { JobMetadata } from '@/components/progress/JobMetadata';
import { OutlierTable } from '@/components/results/OutlierTable';
import { DeleteJobDialog } from '@/components/results/DeleteJobDialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

/**
 * Firestore Timestamp shapes that show up on the client.
 *
 * Firestore serializes Timestamp values differently depending on which SDK /
 * transport produced them:
 *   - Admin SDK REST:   { _seconds, _nanoseconds }
 *   - Client SDK JSON:  { seconds, nanoseconds }
 *   - toJSON() output:  { toDate?: () => Date } on Timestamp instances
 * Plus the usual primitives we might synthesize in tests/UI:
 *   - Date, ISO string, epoch ms number
 */
type FirestoreTimestampLike =
  | Date
  | string
  | number
  | { _seconds: number; _nanoseconds?: number }
  | { seconds: number; nanoseconds?: number }
  | { toDate: () => Date }
  | null
  | undefined;

/**
 * Normalize whatever the job-status endpoint gave us for `createdAt` into a
 * JS Date, or null if it's unparseable. Used for 7-day retention checks.
 */
function toDate(value: FirestoreTimestampLike): Date | null {
  if (value == null) return null;
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }
  if (typeof value === 'number' || typeof value === 'string') {
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  if (typeof value === 'object') {
    // Firestore Timestamp instance (client SDK) exposes .toDate()
    if (typeof (value as { toDate?: unknown }).toDate === 'function') {
      try {
        const d = (value as { toDate: () => Date }).toDate();
        return d instanceof Date && !Number.isNaN(d.getTime()) ? d : null;
      } catch {
        return null;
      }
    }
    // Serialized forms: {_seconds, _nanoseconds} (admin SDK) or
    // {seconds, nanoseconds} (client SDK / JSON).
    const seconds =
      (value as { _seconds?: number })._seconds ??
      (value as { seconds?: number }).seconds;
    if (typeof seconds === 'number') {
      const nanoseconds =
        (value as { _nanoseconds?: number })._nanoseconds ??
        (value as { nanoseconds?: number }).nanoseconds ??
        0;
      return new Date(seconds * 1000 + Math.floor(nanoseconds / 1e6));
    }
  }
  return null;
}

/**
 * Helper to check if job files expired (7-day retention).
 *
 * Returns false if the timestamp cannot be parsed - we prefer to keep the
 * cleanup UI visible rather than lock a user out based on corrupt data.
 */
/**
 * Derive an approximate overall progress percentage from the job status,
 * since the worker doesn't write stageProgress/overallProgress fields.
 * This prevents the bars from showing 0% throughout the entire run.
 */
function deriveOverallProgress(status: string): number {
  switch (status) {
    case 'queued': return 5;
    case 'processing': return 20;
    case 'training': return 50;
    case 'scoring': return 80;
    case 'complete': return 100;
    case 'error': case 'canceled': return 100;
    default: return 0;
  }
}

function isJobExpired(createdAt: FirestoreTimestampLike): boolean {
  const created = toDate(createdAt);
  if (!created) return false;
  const expirationDate = new Date(created);
  expirationDate.setDate(expirationDate.getDate() + 7);
  return new Date() > expirationDate;
}

/**
 * Dedicated progress page at /job/:id route.
 *
 * Features:
 * - Displays job progress with all components from 03-03A
 * - Shows elapsed time, estimated remaining time, file info
 * - Cancel button with confirmation dialog (FE-07)
 * - Polling stops on terminal states (complete/error/canceled)
 * - Terminal state messages displayed
 *
 * Per user decision: Dedicated page (not modal), shows additional info,
 * cancel button with confirmation.
 */
export default function JobProgress() {
  const { id } = useParams();
  const { data: job, isLoading, error } = useJobPolling(id || null);
  const { mutate: cancelJob } = useJobCancellation();

  if (isLoading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (error) return <div className="min-h-screen flex items-center justify-center">Error loading job status</div>;
  if (!job) return <div className="min-h-screen flex items-center justify-center">Job not found</div>;

  const handleCancel = () => {
    cancelJob(id!);
  };

  const isActive = job.status !== 'complete' && job.status !== 'error' && job.status !== 'canceled';

  return (
    <div className="min-h-screen bg-slate-50 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle>Job Progress</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <StageIndicator currentStage={job.status} />

            <DualProgressBar
              stageProgress={job.stageProgress || deriveOverallProgress(job.status)}
              overallProgress={job.overallProgress || deriveOverallProgress(job.status)}
              stageName={job.status.charAt(0).toUpperCase() + job.status.slice(1)}
            />

            <JobMetadata
              // Firestore timestamps can arrive as {_seconds,_nanoseconds}
              // objects; toDate handles that and falls back to "now" so the
              // elapsed-time display doesn't render "Invalid Date".
              startTime={toDate(job.createdAt) ?? new Date()}
              estimatedDuration={15 * 60 * 1000} // 15 minutes
              fileName={job.fileName || 'Unknown'}
              fileSize={job.fileSize || 0}
            />

            {isActive && (
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive">Cancel Job</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Cancel Job?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will stop the job and you won't receive results. This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Keep Running</AlertDialogCancel>
                    <AlertDialogAction onClick={handleCancel}>Cancel Job</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            )}

            {job.status === 'complete' && (
              <>
                <div className="p-4 bg-green-50 border border-green-200 rounded-md">
                  <p className="text-green-800 font-semibold">Job completed successfully!</p>
                  <p className="text-green-700 text-sm mt-1">View results below.</p>
                </div>

                {/* File expiration and download section */}
                <div className="flex items-center gap-3">
                  {/* Expired job message */}
                  {(isJobExpired(job.createdAt) || job.filesExpired) && (
                    <div className="flex-1 p-3 border rounded-lg bg-yellow-50 border-yellow-200">
                      <p className="text-sm text-yellow-800">
                        Files expired - data deleted after 7-day retention period
                      </p>
                    </div>
                  )}

                  {/* Download button - hidden if expired */}
                  {!isJobExpired(job.createdAt) && !job.filesExpired && (
                    <Button onClick={() => window.location.href = `${API_BASE}/api/jobs/${id}/export`}>
                      Download Results CSV
                    </Button>
                  )}

                  {/* Manual delete - show unless already marked expired.
                       Don't gate on isJobExpired: if lifecycle deletion is
                       delayed/misconfigured, users need the cleanup path. */}
                  {!job.filesExpired && (
                    <DeleteJobDialog jobId={id!} />
                  )}
                </div>
              </>
            )}

            {job.status === 'error' && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-800 font-semibold">Job failed</p>
                <p className="text-red-700 text-sm mt-1">{job.error || 'Unknown error occurred'}</p>
                {!job.filesExpired && (
                  <div className="mt-3">
                    <DeleteJobDialog jobId={id!} />
                  </div>
                )}
              </div>
            )}

            {job.status === 'canceled' && (
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-md">
                <p className="text-gray-800 font-semibold">Job was canceled</p>
                {!job.filesExpired && (
                  <div className="mt-3">
                    <DeleteJobDialog jobId={id!} />
                  </div>
                )}
              </div>
            )}

            {job.status === 'complete' && job.outliers && job.outliers.length > 0 && (
              <div className="mt-6">
                <OutlierTable outliers={job.outliers} />
              </div>
            )}

            {job.status === 'complete' && (!job.outliers || job.outliers.length === 0) && (
              <div className="mt-6 p-4 border rounded-lg bg-muted">
                <p className="text-sm text-muted-foreground">
                  No outliers detected in this dataset.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
