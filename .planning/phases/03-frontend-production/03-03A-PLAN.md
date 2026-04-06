---
phase: 03-frontend-production
plan: 03A
type: execute
wave: 2
depends_on: [03-01]
files_modified:
  - frontend/client/hooks/useJobPolling.ts
  - frontend/client/components/progress/StageIndicator.tsx
  - frontend/client/components/progress/DualProgressBar.tsx
autonomous: true
requirements: [FE-08, FE-09, FE-10, FE-04, FE-05]

must_haves:
  truths:
    - "Polling hook stops when job reaches terminal state (complete/error/canceled)"
    - "Polling cleanup prevents memory leaks on component unmount"
    - "User sees multi-stage progress indicator (Queued → Preprocessing → Training → Scoring)"
    - "Progress bar shows both stage percent and overall job percent"
  artifacts:
    - path: "frontend/client/hooks/useJobPolling.ts"
      provides: "TanStack Query polling hook with conditional termination"
      contains: "refetchInterval"
      min_lines: 30
    - path: "frontend/client/components/progress/StageIndicator.tsx"
      provides: "Step badges showing current stage"
      contains: "Badge"
    - path: "frontend/client/components/progress/DualProgressBar.tsx"
      provides: "Dual progress bars for stage and overall progress"
      contains: "Progress"
  key_links:
    - from: "useJobPolling hook"
      to: "/api/job-status/:id endpoint"
      via: "fetch in queryFn"
      pattern: "fetch.*job-status"
---

<objective>
Create reusable progress components and polling hook for job status tracking.

Purpose: Provide foundation for progress page with polling lifecycle management and visual progress components. These components will be assembled into JobProgress page in plan 03-03B.
Output: useJobPolling hook with automatic cleanup, StageIndicator badges, DualProgressBar component.
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

# User decisions (from CONTEXT.md)
User locked decisions:
- Step indicator with badges: Queued → Preprocessing → Training → Scoring
- Progress bar shows BOTH stage percent AND overall job percent
- Polling stops at terminal states, cleanup on unmount
</context>

<interfaces>
<!-- Firestore job status document structure -->
From Phase 02 worker:
```typescript
interface JobStatus {
  jobId: string;
  status: 'queued' | 'preprocessing' | 'training' | 'scoring' | 'complete' | 'error' | 'canceled';
  stageProgress?: number;  // 0-100 percent for current stage
  overallProgress?: number;  // 0-100 percent for entire job
  fileName?: string;
  fileSize?: number;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  error?: string;
  errorType?: string;
}
```

TanStack Query useQuery hook:
```typescript
import { useQuery } from '@tanstack/react-query';

const { data, isLoading, error } = useQuery({
  queryKey: ['jobStatus', jobId],
  queryFn: () => fetch(`/api/jobs/${jobId}`).then(r => r.json()),
  refetchInterval: (query) => {
    const status = query.state.data?.status;
    if (['complete', 'error', 'canceled'].includes(status)) {
      return false; // Stop polling
    }
    return 2000; // Poll every 2 seconds
  },
  enabled: !!jobId
});
```

shadcn/ui components:
```typescript
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create useJobPolling hook with TanStack Query</name>
  <files>frontend/client/hooks/useJobPolling.ts</files>
  <action>
Create polling hook using TanStack Query's refetchInterval with conditional termination (FE-08, FE-09, FE-10).

Pattern (from research Pattern 2):
```typescript
import { useQuery } from '@tanstack/react-query';

export function useJobPolling(jobId: string | null) {
  return useQuery({
    queryKey: ['jobStatus', jobId],
    queryFn: async () => {
      const response = await fetch(`/api/job-status/${jobId}`);
      if (!response.ok) throw new Error('Failed to fetch job status');
      return response.json();
    },
    enabled: !!jobId, // Don't poll if no jobId
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
```

This eliminates stale closures and manual cleanup issues (FE-10).
  </action>
  <verify>
    <automated>grep -q "refetchInterval" frontend/client/hooks/useJobPolling.ts && grep -q "complete.*error.*canceled" frontend/client/hooks/useJobPolling.ts && echo "SUCCESS"</automated>
  </verify>
  <done>useJobPolling.ts created with TanStack Query. refetchInterval function returns false for terminal states. enabled flag prevents polling without jobId. Automatic cleanup on unmount.</done>
</task>

<task type="auto">
  <name>Task 2: Create StageIndicator component with badges</name>
  <files>frontend/client/components/progress/StageIndicator.tsx</files>
  <action>
Create step indicator showing job stages with Badge components (FE-04).

Stages (from user decision):
1. Queued
2. Preprocessing
3. Training
4. Scoring

Badge variants:
- Completed stages: "default" variant with checkmark
- Current stage: "secondary" variant (highlighted)
- Upcoming stages: "outline" variant

Pattern (from research code example):
```typescript
import { Badge } from "@/components/ui/badge";

interface StageIndicatorProps {
  currentStage: string;
}

const STAGES = ['queued', 'preprocessing', 'training', 'scoring'];
const STAGE_LABELS = {
  queued: 'Queued',
  preprocessing: 'Preprocessing',
  training: 'Training',
  scoring: 'Scoring'
};

export function StageIndicator({ currentStage }: StageIndicatorProps) {
  const currentIndex = STAGES.indexOf(currentStage);

  return (
    <div className="flex items-center gap-2">
      {STAGES.map((stage, idx) => (
        <Badge
          key={stage}
          variant={
            idx < currentIndex ? "default" :  // Completed
            idx === currentIndex ? "secondary" :  // Current
            "outline"  // Upcoming
          }
        >
          {idx < currentIndex && "✓ "}{STAGE_LABELS[stage]}
        </Badge>
      ))}
    </div>
  );
}
```
  </action>
  <verify>
    <automated>grep -q "Badge" frontend/client/components/progress/StageIndicator.tsx && grep -q "queued.*preprocessing.*training.*scoring" frontend/client/components/progress/StageIndicator.tsx && echo "SUCCESS"</automated>
  </verify>
  <done>StageIndicator.tsx created with Badge components. Four stages rendered (Queued, Preprocessing, Training, Scoring). Completed stages show checkmark. Current stage uses secondary variant. Upcoming stages use outline variant.</done>
</task>

<task type="auto">
  <name>Task 3: Create DualProgressBar component</name>
  <files>frontend/client/components/progress/DualProgressBar.tsx</files>
  <action>
Create dual progress bar showing stage percent AND overall percent (FE-05).

Pattern (from research code example):
```typescript
import { Progress } from "@/components/ui/progress";

interface DualProgressBarProps {
  stageProgress: number;
  overallProgress: number;
  stageName: string;
}

export function DualProgressBar({ stageProgress, overallProgress, stageName }: DualProgressBarProps) {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-sm text-gray-600 mb-1">
          {stageName}: {stageProgress}%
        </p>
        <Progress value={stageProgress} className="h-2" />
      </div>
      <div>
        <p className="text-sm text-gray-600 mb-1">
          Overall: {overallProgress}%
        </p>
        <Progress value={overallProgress} className="h-3" />
      </div>
    </div>
  );
}
```

Per user decision: "Training (65% of this stage)" + "Overall: 75% complete"
  </action>
  <verify>
    <automated>grep -q "stageProgress" frontend/client/components/progress/DualProgressBar.tsx && grep -q "overallProgress" frontend/client/components/progress/DualProgressBar.tsx && echo "SUCCESS"</automated>
  </verify>
  <done>DualProgressBar.tsx created with two Progress components. Stage progress shows current stage percent. Overall progress shows job completion percent. Both display numeric percentages.</done>
</task>

</tasks>

<verification>
Run all automated task verifications:
1. useJobPolling hook uses refetchInterval with terminal state checks
2. StageIndicator renders badges for all stages
3. DualProgressBar shows stage and overall progress

Component isolation tests:
```typescript
// Test StageIndicator with different stages
<StageIndicator currentStage="training" />  // Should highlight "Training"

// Test DualProgressBar with sample data
<DualProgressBar stageProgress={65} overallProgress={75} stageName="Training" />

// Test useJobPolling with mock jobId
const { data } = useJobPolling('test-job-id');
```
</verification>

<success_criteria>
- [x] useJobPolling hook created with TanStack Query refetchInterval
- [x] Polling stops at terminal states (complete/error/canceled)
- [x] Polling cleanup automatic on unmount (no memory leaks)
- [x] StageIndicator shows 4 stages with Badge variants
- [x] DualProgressBar displays stage and overall percentages
- [x] All components TypeScript-typed with explicit props interfaces
</success_criteria>

<output>
After completion, create `.planning/phases/03-frontend-production/03-03A-SUMMARY.md`
</output>
