import { Progress } from "@/components/ui/progress";

interface Contribution {
  column: string;
  percentage: number;
  // The respondent's actual answer for this column, and what the model
  // reconstructed instead. A gap between the two is what pushed this row's
  // reconstruction error up. Optional for backward compatibility with jobs
  // scored before these fields were added.
  actual?: string | number | null;
  predicted?: string | number | null;
}

interface ContributionScoresProps {
  contributions: Contribution[];
}

function formatValue(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

/**
 * Displays per-column contribution scores as horizontal bar chart.
 *
 * Per plan 04-05:
 * - Shows all columns sorted by contribution (already sorted from backend)
 * - Color-coded by contribution level (high=red/orange, medium=yellow, low=gray)
 * - Displays percentage with 1 decimal place
 *
 * Color thresholds (Claude's discretion):
 * - >20% = red (high contribution)
 * - >10% = orange (medium-high)
 * - >5% = yellow (medium)
 * - <=5% = gray (low contribution)
 */
export function ContributionScores({ contributions }: ContributionScoresProps) {
  return (
    <div className="space-y-2 py-2">
      <h4 className="text-sm font-semibold mb-1">Per-Column Contributions</h4>
      <p className="text-xs text-muted-foreground mb-3">
        Each column's share of this row's reconstruction error, with the actual
        answer vs. what the model expected.
      </p>
      <div className="space-y-3">
        {contributions.map(({ column, percentage, actual, predicted }) => {
          const hasComparison = actual !== undefined || predicted !== undefined;
          const mismatch =
            hasComparison && formatValue(actual) !== formatValue(predicted);
          return (
            <div key={column} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="font-medium">{column}</span>
                <span className="text-muted-foreground">{percentage.toFixed(1)}%</span>
              </div>
              <Progress
                value={percentage}
                // Color the *filled* indicator rather than the track; the
                // Progress root's className only styles the background track,
                // which left every bar looking the same regardless of
                // contribution level.
                indicatorClassName={getContributionColorClass(percentage)}
              />
              {hasComparison && (
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
                  <span className="text-muted-foreground">actual</span>
                  <span
                    className={`font-mono ${mismatch ? "text-red-600 font-medium" : "text-foreground"}`}
                  >
                    {formatValue(actual)}
                  </span>
                  <span className="text-muted-foreground" aria-hidden>
                    →
                  </span>
                  <span className="text-muted-foreground">predicted</span>
                  <span className="font-mono text-foreground">{formatValue(predicted)}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Returns color class based on contribution percentage threshold.
 *
 * Thresholds chosen to highlight high contributors while keeping
 * visual hierarchy clear for researchers.
 */
function getContributionColorClass(percentage: number): string {
  if (percentage > 20) return "bg-red-500";      // High contribution
  if (percentage > 10) return "bg-orange-500";   // Medium-high
  if (percentage > 5) return "bg-yellow-500";    // Medium
  return "bg-gray-400";                          // Low contribution
}
