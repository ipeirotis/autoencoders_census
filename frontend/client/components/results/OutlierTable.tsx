import { useState } from "react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight } from "lucide-react";
import { ContributionScores } from "./ContributionScores";

interface Outlier {
  rowId: number;
  score: number;
  contributions: Array<{
    column: string;
    percentage: number;
  }>;
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
 * - Score displayed with 3 decimal places (precision for outlier threshold)
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
        {outliers.map((outlier) => (
          <OutlierRow key={outlier.rowId} outlier={outlier} />
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
function OutlierRow({ outlier }: { outlier: Outlier }) {
  const [isOpen, setIsOpen] = useState(false);

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
            <span className="text-sm font-medium">Row {outlier.rowId}</span>
            <span className="text-sm text-muted-foreground">
              Score: {outlier.score.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Expanded details */}
        <CollapsibleContent className="px-4 pb-4 bg-slate-50">
          <ContributionScores contributions={outlier.contributions} />
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
