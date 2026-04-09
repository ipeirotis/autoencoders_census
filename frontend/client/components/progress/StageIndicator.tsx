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
  const STAGES = ['queued', 'processing', 'training', 'scoring'] as const;
  const STAGE_LABELS: Record<typeof STAGES[number], string> = {
    queued: 'Queued',
    processing: 'Processing',
    training: 'Training',
    scoring: 'Scoring',
  };

  // Terminal states mean all stages are done; indexOf returns -1 for these,
  // so treat them as "past the last stage" so every badge shows completed.
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
