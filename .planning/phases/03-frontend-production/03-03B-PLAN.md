---
phase: 03-frontend-production
plan: 03B
type: execute
wave: 3
depends_on: [03-01, 03-03A]
files_modified:
  - frontend/client/pages/JobProgress.tsx
  - frontend/client/components/progress/JobMetadata.tsx
  - frontend/client/hooks/useJobCancellation.ts
  - frontend/client/App.tsx
autonomous: true
requirements: [FE-06, FE-07]

must_haves:
  truths:
    - "User can navigate to /job/:id progress page after upload"
    - "Progress page displays elapsed time and estimated remaining time"
    - "User can cancel job from progress page with confirmation dialog"
    - "Progress page assembles all components from 03-03A"
  artifacts:
    - path: "frontend/client/pages/JobProgress.tsx"
      provides: "Dedicated /job/:id progress page"
      min_lines: 80
    - path: "frontend/client/components/progress/JobMetadata.tsx"
      provides: "Elapsed time and file info display"
      contains: "formatDuration"
    - path: "frontend/client/hooks/useJobCancellation.ts"
      provides: "Job cancellation mutation"
      contains: "useMutation"
  key_links:
    - from: "frontend/client/pages/JobProgress.tsx"
      to: "useJobPolling hook"
      via: "hook invocation"
      pattern: "useJobPolling"
    - from: "frontend/client/App.tsx"
      to: "JobProgress page"
      via: "react-router Route"
      pattern: 'Route.*path="/job/:id"'
---

<objective>
Assemble progress page with metadata display, job cancellation, and routing.

Purpose: Complete progress tracking feature by composing components from 03-03A into functional page with cancellation capability and routing integration.
Output: JobProgress page at /job/:id with elapsed/remaining time, file info, cancel button, and confirmation dialog.
</objective>

<execution_context>
@/Users/aaron/.claude/get-shit-done/workflows/execute-plan.md
@/Users/aaron/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/03-frontend-production/03-CONTEXT.md
@.planning/phases/03-frontend-production/03-RESEARCH.md
@.planning/phases/03-frontend-production/03-03A-SUMMARY.md

# User decisions (from CONTEXT.md)
User locked decisions:
- Dedicated progress page at /job/:id route (not modal or inline)
- Additional info: Elapsed time, Estimated time remaining, File name & size
- Cancel button with confirmation dialog
</context>

<interfaces>
<!-- Components from 03-03A -->
From 03-03A-SUMMARY.md:
```typescript
// useJobPolling hook
export function useJobPolling(jobId: string | null): {
  data: JobStatus | undefined;
  isLoading: boolean;
  error: Error | null;
}

// StageIndicator component
export function StageIndicator({ currentStage }: { currentStage: string }): JSX.Element

// DualProgressBar component
export function DualProgressBar({
  stageProgress,
  overallProgress,
  stageName
}: {
  stageProgress: number;
  overallProgress: number;
  stageName: string;
}): JSX.Element
```

TanStack Query mutation:
```typescript
import { useMutation } from '@tanstack/react-query';

const mutation = useMutation({
  mutationFn: (jobId: string) => fetch(`/api/jobs/${jobId}`, { method: 'DELETE' }),
  onSuccess: () => { /* ... */ }
});
```

shadcn/ui components:
```typescript
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
```

React Router (from 03-01):
```typescript
import { BrowserRouter, Routes, Route, useParams } from 'react-router-dom';
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create JobMetadata component for elapsed/remaining time and file info</name>
  <files>frontend/client/components/progress/JobMetadata.tsx</files>
  <action>
Create metadata display showing elapsed time, estimated remaining time, file name, and file size (from user decision in CONTEXT.md).

```typescript
import { useState, useEffect } from 'react';

interface JobMetadataProps {
  startTime: Date;
  estimatedDuration?: number; // milliseconds
  fileName: string;
  fileSize: number;
}

export function JobMetadata({ startTime, estimatedDuration, fileName, fileSize }: JobMetadataProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTime.getTime());
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime]);

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const remaining = estimatedDuration ? estimatedDuration - elapsed : null;

  return (
    <div className="text-sm text-gray-600 space-y-1">
      <p>Elapsed: {formatDuration(elapsed)}</p>
      {remaining && remaining > 0 && (
        <p>Estimated remaining: ~{formatDuration(remaining)}</p>
      )}
      <p>File: {fileName} ({formatFileSize(fileSize)})</p>
    </div>
  );
}
```

Elapsed time updates every second. Estimated remaining calculated from estimatedDuration - elapsed.
  </action>
  <verify>
    <automated>grep -q "formatDuration" frontend/client/components/progress/JobMetadata.tsx && grep -q "formatFileSize" frontend/client/components/progress/JobMetadata.tsx && echo "SUCCESS"</automated>
  </verify>
  <done>JobMetadata.tsx created with elapsed time counter, estimated remaining time, file name, and file size. Elapsed updates every second via useEffect + setInterval. File size formatted in B/KB/MB.</done>
</task>

<task type="auto">
  <name>Task 2: Create useJobCancellation hook</name>
  <files>frontend/client/hooks/useJobCancellation.ts</files>
  <action>
Create hook for job cancellation with confirmation (FE-06, FE-07).

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useToast } from "@/components/ui/use-toast";

export function useJobCancellation() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (jobId: string) => {
      const response = await fetch(`/api/jobs/${jobId}`, {
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
```

Hook returns mutation with mutate function for cancellation. Confirmation dialog handled in JobProgress page component.
  </action>
  <verify>
    <automated>grep -q "useMutation" frontend/client/hooks/useJobCancellation.ts && grep -q "DELETE" frontend/client/hooks/useJobCancellation.ts && echo "SUCCESS"</automated>
  </verify>
  <done>useJobCancellation.ts created with TanStack Query useMutation. Sends DELETE request to /api/jobs/:id. Invalidates query cache on success. Shows toast notifications for success/error.</done>
</task>

<task type="auto">
  <name>Task 3: Create JobProgress page with routing</name>
  <files>frontend/client/pages/JobProgress.tsx, frontend/client/App.tsx</files>
  <action>
Create dedicated progress page at /job/:id route (per user decision). Compose all components from 03-03A.

Add react-router-dom routing to App.tsx:
```typescript
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import JobProgress from './pages/JobProgress';
import Index from './pages/Index';

<BrowserRouter>
  <Routes>
    <Route path="/" element={<Index />} />
    <Route path="/job/:id" element={<JobProgress />} />
  </Routes>
</BrowserRouter>
```

Create JobProgress.tsx:
```typescript
import { useParams } from 'react-router-dom';
import { useJobPolling } from '@/hooks/useJobPolling';
import { useJobCancellation } from '@/hooks/useJobCancellation';
import { StageIndicator } from '@/components/progress/StageIndicator';
import { DualProgressBar } from '@/components/progress/DualProgressBar';
import { JobMetadata } from '@/components/progress/JobMetadata';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function JobProgress() {
  const { id } = useParams();
  const { data: job, isLoading, error } = useJobPolling(id || null);
  const { mutate: cancelJob } = useJobCancellation();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error loading job status</div>;
  if (!job) return <div>Job not found</div>;

  const handleCancel = () => {
    cancelJob(id!);
  };

  return (
    <div className="container mx-auto p-8">
      <Card>
        <CardHeader>
          <CardTitle>Job Progress</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <StageIndicator currentStage={job.status} />
          <DualProgressBar
            stageProgress={job.stageProgress || 0}
            overallProgress={job.overallProgress || 0}
            stageName={job.status}
          />
          <JobMetadata
            startTime={new Date(job.createdAt)}
            estimatedDuration={15 * 60 * 1000} // 15 minutes
            fileName={job.fileName || 'Unknown'}
            fileSize={job.fileSize || 0}
          />

          {job.status !== 'complete' && job.status !== 'error' && job.status !== 'canceled' && (
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

          {job.status === 'complete' && <div>Job completed! View results below.</div>}
          {job.status === 'error' && <div>Job failed: {job.error}</div>}
          {job.status === 'canceled' && <div>Job was canceled.</div>}
        </CardContent>
      </Card>
    </div>
  );
}
```

User navigates to /job/:id after upload completes. Cancel button only shown for active jobs (not complete/error/canceled). Confirmation dialog prevents accidental cancellation (FE-07).
  </action>
  <verify>
    <automated>grep -q "useParams" frontend/client/pages/JobProgress.tsx && grep -q "BrowserRouter" frontend/client/App.tsx && echo "SUCCESS"</automated>
  </verify>
  <done>JobProgress.tsx created with all progress components. App.tsx configured with react-router-dom Routes. /job/:id route renders JobProgress page. Cancel button shows AlertDialog confirmation. Polling hook provides real-time updates.</done>
</task>

</tasks>

<verification>
Run all automated task verifications:
1. JobMetadata formats elapsed time and file size
2. useJobCancellation hook uses DELETE mutation
3. JobProgress page uses useParams and routing configured

Manual verification (requires running app):
1. Navigate to /job/:id with real job ID
2. Verify all components render: stage indicator, progress bars, metadata, cancel button
3. Verify elapsed time increments every second
4. Click Cancel Job → confirm dialog appears
5. Verify polling stops when job completes
6. Navigate away and back → no memory leaks in console

Integration test:
1. Upload CSV via Dropzone
2. After upload, app navigates to /job/:id
3. Progress page displays with all info
4. Poll updates reflect status changes
5. Job completion shows results
</verification>

<success_criteria>
- [x] JobMetadata shows elapsed time, estimated remaining, file info
- [x] Elapsed time updates every second
- [x] useJobCancellation hook sends DELETE request
- [x] JobProgress page created at /job/:id route
- [x] App.tsx configured with react-router-dom
- [x] Cancel button shows confirmation AlertDialog
- [x] All components from 03-03A integrated into page
- [x] Page displays terminal state messages (complete/error/canceled)
</success_criteria>

<output>
After completion, create `.planning/phases/03-frontend-production/03-03B-SUMMARY.md`
</output>
