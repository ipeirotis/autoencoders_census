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

  // When the job has reached a terminal 'complete' state, every stage has
  // finished - mark them all as completed rather than leaving the indicator
  // blank (currentIndex would otherwise be -1 for terminal statuses).
  const isComplete = currentStage === 'complete';
  const currentIndex = isComplete
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
