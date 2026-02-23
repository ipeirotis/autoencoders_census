/**
 * Renders CSV data as a table 
 */

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface PreviewTableProps {
  rows: Record<string, unknown>[];
  headers: string[];
  totalRows: number;
}

export const PreviewTable: React.FC<PreviewTableProps> = ({
  rows,
  headers,
  totalRows,
}) => {
  const [showAllRows, setShowAllRows] = useState(false);
  const [showAllColumns, setShowAllColumns] = useState(false);

  const displayRows = showAllRows ? rows : rows.slice(0, 20);
  const displayHeaders = showAllColumns ? headers : headers.slice(0, 10);
  const hasMoreColumns = headers.length > 10;
  const hasMoreRows = rows.length > 20;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-600">
          Showing first {displayRows.length} of {rows.length} rows
          {hasMoreColumns && ` (showing ${displayHeaders.length} of ${headers.length} columns)`}
        </p>
        <div className="flex gap-2">
          {hasMoreRows && (
            <Button
              onClick={() => setShowAllRows(!showAllRows)}
              variant="outline"
              size="sm"
              className="text-xs"
            >
              {showAllRows ? "Show Less Rows" : "Show All Rows"}
            </Button>
          )}
          {hasMoreColumns && (
            <Button
              onClick={() => setShowAllColumns(!showAllColumns)}
              variant="outline"
              size="sm"
              className="text-xs"
            >
              {showAllColumns ? "Show Less Columns" : "Show All Columns"}
            </Button>
          )}
        </div>
      </div>
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table
          data-testid="preview-table"
          className="w-full text-sm border-collapse"
        >
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {displayHeaders.map((header, index) => (
                <th
                  key={`header-${index}-${header}`}
                  className="px-4 py-2 text-left font-semibold text-gray-900 whitespace-nowrap"
                >
                  {header}
                </th>
              ))}
              {!showAllColumns && hasMoreColumns && (
                <th key="more-columns" className="px-4 py-2 text-left font-semibold text-gray-600 italic whitespace-nowrap">
                  +{headers.length - 10} more
                </th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {displayRows.map((row, rowIndex) => (
              <tr
                key={`row-${rowIndex}`}
                className={cn(
                  rowIndex % 2 === 0 ? "bg-white" : "bg-gray-50",
                  "hover:bg-blue-50 transition-colors"
                )}
              >
                {displayHeaders.map((header, colIndex) => (
                  <td
                    key={`cell-${rowIndex}-${colIndex}-${header}`}
                    className="px-4 py-2 text-gray-700"
                  >
                    <span className="block truncate max-w-xs">
                      {/* CONDITIONAL FORMATTING START */}
                      {header === "reconstruction_error" ? (
                        <span className="font-bold text-red-600">
                          {typeof row[header] === "number"
                            ? (row[header] as number).toFixed(4)
                            : String(row[header])}
                        </span>
                      ) : (
                        String(row[header] || "")
                      )}
                      {/* CONDITIONAL FORMATTING END */}
                    </span>
                  </td>
                ))}
                {!showAllColumns && hasMoreColumns && (
                  <td key={`more-${rowIndex}`} className="px-4 py-2 text-gray-500 italic">...</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
