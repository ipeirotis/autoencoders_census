import { Badge } from "@/components/ui/badge";

interface StageIndicatorProps {
  currentStage: string;
}

/**
 * Stage indicator showing job processing stages with Badge components.
 *
 * Displays four stages: Queued → Preprocessing → Training → Scoring
 * - Completed stages: default variant with checkmark
 * - Current stage: secondary variant (highlighted)
 * - Upcoming stages: outline variant
 *
 * Implements FE-04 (multi-stage progress indicator).
 */
export function StageIndicator({ currentStage }: StageIndicatorProps) {
  const STAGES = ['queued', 'preprocessing', 'training', 'scoring'] as const;
  const STAGE_LABELS: Record<typeof STAGES[number], string> = {
    queued: 'Queued',
    preprocessing: 'Preprocessing',
    training: 'Training',
    scoring: 'Scoring',
  };

  const currentIndex = STAGES.indexOf(currentStage as typeof STAGES[number]);

  return (
    <div className="flex items-center gap-2">
      {STAGES.map((stage, idx) => {
        const isCompleted = idx < currentIndex;
        const isCurrent = idx === currentIndex;
        const isUpcoming = idx > currentIndex;

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
