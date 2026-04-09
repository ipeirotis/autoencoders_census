import { useParams, useNavigate } from 'react-router-dom';
import { useJobPolling } from '@/hooks/useJobPolling';
import { useJobCancellation } from '@/hooks/useJobCancellation';
import { StageIndicator } from '@/components/progress/StageIndicator';
import { DualProgressBar } from '@/components/progress/DualProgressBar';
import { JobMetadata } from '@/components/progress/JobMetadata';
import { PreviewTable } from '@/components/PreviewTable';
import { PreviewErrorBoundary } from '@/components/error-boundaries/PreviewErrorBoundary';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

/**
 * Dedicated progress + results page at /job/:id route.
 *
 * While the job is active, shows progress indicators, elapsed time, and a
 * cancel button. Once the job completes, renders the outlier results table
 * and dropped-columns panel inline — this is the primary results view since
 * Index.tsx navigates here after uploadCsv() succeeds.
 */
export default function JobProgress() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { data: job, isLoading, error } = useJobPolling(id || null);
  const { mutate: cancelJob } = useJobCancellation();

  if (isLoading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (error) return <div className="min-h-screen flex items-center justify-center">Error loading job status</div>;
  if (!job) return <div className="min-h-screen flex items-center justify-center">Job not found</div>;

  const handleCancel = () => {
    cancelJob(id!);
  };

  const isActive = job.status !== 'complete' && job.status !== 'error' && job.status !== 'canceled';
  const outliers = job.outliers || [];
  const stats = job.stats;

  /** Put reconstruction_error first so the most important column is visible. */
  const getOrderedHeaders = (row: Record<string, any> | undefined) => {
    if (!row) return [];
    const allHeaders = Object.keys(row);
    const dataHeaders = allHeaders.filter(h => h !== 'reconstruction_error');
    return ['reconstruction_error', ...dataHeaders];
  };

  return (
    <div className="min-h-screen bg-slate-50 py-8 px-4">
      <div className={job.status === 'complete' && outliers.length > 0 ? "max-w-5xl mx-auto" : "max-w-3xl mx-auto"}>

        {/* Progress Card — shown while active or on error/canceled */}
        {(isActive || job.status === 'error' || job.status === 'canceled') && (
          <Card>
            <CardHeader>
              <CardTitle>Job Progress</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <StageIndicator currentStage={job.status} />

              <DualProgressBar
                stageProgress={job.stageProgress || 0}
                overallProgress={job.overallProgress || 0}
                stageName={job.status.charAt(0).toUpperCase() + job.status.slice(1)}
              />

              <JobMetadata
                startTime={new Date(job.createdAt)}
                estimatedDuration={15 * 60 * 1000}
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

              {job.status === 'error' && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-red-800 font-semibold">Job failed</p>
                  <p className="text-red-700 text-sm mt-1">{job.error || 'Unknown error occurred'}</p>
                  <Button onClick={() => navigate('/')} variant="outline" className="mt-3">
                    Try Again
                  </Button>
                </div>
              )}

              {job.status === 'canceled' && (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-md">
                  <p className="text-gray-800 font-semibold">Job was canceled</p>
                  <Button onClick={() => navigate('/')} variant="outline" className="mt-3">
                    Start New Analysis
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Results Section — shown when job completes with outliers */}
        {job.status === 'complete' && outliers.length > 0 && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

              {/* Left Column: Outlier Table (75% width) */}
              <div className="lg:col-span-3 bg-white rounded-2xl shadow p-6">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-green-700">Analysis Complete</h2>
                    <p className="text-gray-600">
                      Found {outliers.length} outliers.
                      {' '}Kept {stats?.kept_columns?.length || 0} columns.
                    </p>
                  </div>
                  <Button onClick={() => navigate('/')} variant="outline">Analyze New File</Button>
                </div>

                <div className="mt-6">
                  <h3 className="text-lg font-semibold mb-3">Top Outliers</h3>
                  <PreviewErrorBoundary>
                    <PreviewTable
                      rows={outliers}
                      headers={getOrderedHeaders(outliers[0])}
                      totalRows={outliers.length}
                    />
                  </PreviewErrorBoundary>
                </div>
              </div>

              {/* Right Column: Dropped Columns Panel (25% width) */}
              <div className="lg:col-span-1 bg-white rounded-2xl shadow p-6 h-fit max-h-[80vh] flex flex-col">
                <div className="mb-4">
                  <h3 className="font-bold text-gray-800 text-lg">Dropped Columns</h3>
                  <p className="text-xs text-gray-500">
                    Removed due to high cardinality ({'>'}9) or being a single value.
                  </p>
                </div>

                <div className="overflow-y-auto pr-2 space-y-2 flex-1">
                  {!stats?.ignored_columns?.length ? (
                    <p className="text-sm text-gray-400 italic">No columns dropped.</p>
                  ) : (
                    stats.ignored_columns.map((col, idx) => (
                      <div key={idx} className="p-3 bg-slate-50 rounded border border-slate-200 text-sm group hover:border-red-300 transition-colors">
                        <div className="font-semibold text-slate-700 truncate" title={col.name}>
                          {col.name}
                        </div>
                        <div className="flex justify-between items-center mt-1">
                          <span className="text-xs text-slate-500">Unique Values:</span>
                          <span className="text-xs font-mono bg-slate-200 px-1.5 py-0.5 rounded text-slate-600">
                            {col.unique_values}
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Completion with no outliers */}
        {job.status === 'complete' && outliers.length === 0 && (
          <Card>
            <CardContent className="p-12 text-center">
              <h2 className="text-2xl font-bold text-green-700 mb-2">Analysis Complete</h2>
              <p className="text-gray-600 mb-4">No outliers were detected in the dataset.</p>
              <Button onClick={() => navigate('/')}>Analyze Another File</Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
