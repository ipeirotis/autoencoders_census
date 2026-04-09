import { useState } from "react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight } from "lucide-react";
import { ContributionScores } from "./ContributionScores";

/**
 * Outlier record shape as written by worker.py.
 *
 * Each record is a flat dict containing:
 *   - all original CSV row values (string keys, arbitrary values)
 *   - reconstruction_error: number (the outlier score, added by the worker)
 *   - __meta: reserved namespace for worker-added metadata that must not
 *     collide with user-uploaded column names (contributions, etc.)
 */
interface Outlier {
  reconstruction_error: number;
  __meta?: {
    contributions?: Array<{
      column: string;
      percentage: number;
    }>;
  };
  // Original CSV columns are also present as additional keys
  [key: string]: unknown;
}

interface OutlierTableProps {
  outliers: Outlier[];
}

/**
 * Table component displaying outliers with expandable row details.
 *
 * Per plan 04-05:
 * - Expandable row pattern similar to GitHub PR file list
 * - Chevron icon indicates expand/collapse state
 * - ContributionScores component renders in expanded panel
 * - Reconstruction error displayed with 3 decimal places
 * - All outliers displayed (no pagination for v1)
 *
 * Collapsible component provides automatic transition animation.
 */
export function OutlierTable({ outliers }: OutlierTableProps) {
  return (
    <div className="border rounded-lg">
      <div className="bg-muted px-4 py-2 font-semibold text-sm">
        Detected Outliers ({outliers.length})
      </div>
      <div className="divide-y">
        {outliers.map((outlier, index) => (
          <OutlierRow key={index} rowIndex={index} outlier={outlier} />
        ))}
      </div>
    </div>
  );
}

/**
 * Individual outlier row with expandable contribution details.
 *
 * Manages its own expand/collapse state.
 */
function OutlierRow({ outlier, rowIndex }: { outlier: Outlier; rowIndex: number }) {
  const [isOpen, setIsOpen] = useState(false);

  // Defensive: reconstruction_error may be missing if the backend payload
  // shape changes. Show "—" rather than crash with toFixed on undefined.
  const score =
    typeof outlier.reconstruction_error === "number"
      ? outlier.reconstruction_error.toFixed(3)
      : "—";

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div>
        {/* Main row - clickable to expand */}
        <div className="flex items-center px-4 py-3 hover:bg-slate-50 cursor-pointer">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="p-0 h-auto">
              {isOpen ? (
                <ChevronDown className="h-4 w-4 mr-2" />
              ) : (
                <ChevronRight className="h-4 w-4 mr-2" />
              )}
            </Button>
          </CollapsibleTrigger>
          <div className="flex-1 flex items-center justify-between">
            <span className="text-sm font-medium">Row {rowIndex + 1}</span>
            <span className="text-sm text-muted-foreground">
              Reconstruction error: {score}
            </span>
          </div>
        </div>

        {/* Expanded details */}
        <CollapsibleContent className="px-4 pb-4 bg-slate-50">
          <ContributionScores
            contributions={outlier.__meta?.contributions ?? []}
          />
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
