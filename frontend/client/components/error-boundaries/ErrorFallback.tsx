import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { AlertCircle, ChevronDown, Copy, Check } from "lucide-react";
import { FallbackProps } from "react-error-boundary";

/**
 * Shared error fallback UI component with expandable technical details.
 *
 * Features:
 * - "Something went wrong" message with reload button
 * - Expandable technical details (error message, stack trace, timestamp)
 * - Copy to clipboard for bug reporting
 *
 * Used by all error boundaries (root and component-level).
 */
export const ErrorFallback: React.FC<FallbackProps> = ({ error, resetErrorBoundary }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const timestamp = new Date().toISOString();

  const copyErrorDetails = () => {
    const errorDetails = `
Error: ${error.message}
Timestamp: ${timestamp}
Component Stack: ${error.stack || "No stack trace available"}
    `.trim();

    navigator.clipboard.writeText(errorDetails);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card className="max-w-2xl w-full border-red-200 bg-red-50">
      <CardHeader>
        <div className="flex items-start gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-1" />
          <div className="flex-1">
            <CardTitle className="text-red-900">Something went wrong</CardTitle>
            <p className="text-sm text-red-700 mt-2">
              An unexpected error occurred. Please try reloading the page.
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Button
            onClick={resetErrorBoundary}
            variant="default"
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            Reload Page
          </Button>
          <Button
            onClick={copyErrorDetails}
            variant="outline"
            className="gap-2 border-red-300 text-red-700 hover:bg-red-100"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                Copy Error Details
              </>
            )}
          </Button>
        </div>

        <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
          <CollapsibleTrigger className="flex items-center gap-2 text-sm text-red-700 hover:text-red-900">
            <ChevronDown
              className={`w-4 h-4 transition-transform ${isExpanded ? "rotate-180" : ""}`}
            />
            {isExpanded ? "Hide Details" : "Show Details"}
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 p-4 bg-red-100 rounded-lg border border-red-200">
            <div className="space-y-3 text-xs font-mono">
              <div>
                <p className="font-semibold text-red-900 mb-1">Error Message:</p>
                <p className="text-red-800">{error.message}</p>
              </div>
              <div>
                <p className="font-semibold text-red-900 mb-1">Timestamp:</p>
                <p className="text-red-800">{timestamp}</p>
              </div>
              <div>
                <p className="font-semibold text-red-900 mb-1">Stack Trace:</p>
                <pre className="text-red-800 whitespace-pre-wrap break-words">
                  {error.stack || "No stack trace available"}
                </pre>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
};
