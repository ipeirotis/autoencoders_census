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
 */
export function StageIndicator({ currentStage }: StageIndicatorProps) {
  const STAGES = ['queued', 'processing', 'training', 'scoring'] as const;
  const STAGE_LABELS: Record<typeof STAGES[number], string> = {
    queued: 'Queued',
    processing: 'Processing',
    training: 'Training',
    scoring: 'Scoring',
  };

  const currentIndex = STAGES.indexOf(currentStage as typeof STAGES[number]);

  return (
    <div className="flex items-center gap-2">
      {STAGES.map((stage, idx) => {
        const isCompleted = idx < currentIndex;
        const isCurrent = idx === currentIndex;

        return (
          <Badge
            key={stage}
            variant={
              isCompleted ? "default" :
              isCurrent ? "secondary" :
              "outline"
            }
          >
            {isCompleted && "✓ "}{STAGE_LABELS[stage]}
          </Badge>
        );
      })}
    </div>
  );
}
