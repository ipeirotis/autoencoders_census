import { Badge } from "@/components/ui/badge";

interface StageIndicatorProps {
  currentStage: string;
}

/**
 * Stage indicator showing job processing stages with Badge components.
 *
 * Displays four stages: Queued → Processing → Training → Scoring
 * - Completed stages: default variant with checkmark
 * - Current stage: secondary variant (highlighted)
 * - Upcoming stages: outline variant
 *
 * Stage names must match the worker's JobStatus enum in worker.py
 * ('queued', 'processing', 'training', 'scoring', 'complete', 'error',
 * 'canceled'); otherwise indexOf returns -1 and no badge is highlighted.
 *
 * Implements FE-04 (multi-stage progress indicator).
 */
export function StageIndicator({ currentStage }: StageIndicatorProps) {
  const STAGES = ['queued', 'processing', 'training', 'scoring'] as const;
  const STAGE_LABELS: Record<typeof STAGES[number], string> = {
    queued: 'Queued',
    processing: 'Processing',
    training: 'Training',
    scoring: 'Scoring',
  };

  // Terminal states (complete/error/canceled) all imply every stage is
  // finished from the indicator's point of view - the dedicated success/
  // error/canceled cards in JobProgress.tsx convey *which* terminal state
  // was reached. Without this, indexOf would return -1 for those statuses
  // and every stage would render as still upcoming, which is misleading
  // on failed/canceled jobs.
  const TERMINAL_STATES = new Set(['complete', 'error', 'canceled']);
  const isTerminal = TERMINAL_STATES.has(currentStage);
  const currentIndex = isTerminal
    ? STAGES.length
    : STAGES.indexOf(currentStage as typeof STAGES[number]);

  return (
    <div className="flex items-center gap-2">
      {STAGES.map((stage, idx) => {
        const isCompleted = idx < currentIndex;
        const isCurrent = idx === currentIndex;

        return (
          <Badge
            key={stage}
            variant={
              isCompleted ? "default" :  // Completed stages
              isCurrent ? "secondary" :  // Current stage
              "outline"  // Upcoming stages
            }
          >
            {isCompleted && "✓ "}{STAGE_LABELS[stage]}
          </Badge>
        );
      })}
    </div>
  );
}
