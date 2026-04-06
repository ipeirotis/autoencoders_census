import { Progress } from "@/components/ui/progress";

interface DualProgressBarProps {
  stageProgress: number;
  overallProgress: number;
  stageName: string;
}

/**
 * Dual progress bar component showing both stage and overall job progress.
 *
 * Displays two progress bars:
 * - Stage progress: Current stage completion (e.g., "Training: 65%")
 * - Overall progress: Entire job completion (e.g., "Overall: 75%")
 *
 * Per user decision: Shows BOTH stage percent AND overall job percent.
 * Implements FE-05 (dual progress visualization).
 */
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
