import React, { ReactNode } from "react";
import { ErrorBoundary, FallbackProps } from "react-error-boundary";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";

/**
 * Inline error fallback for Results component crashes.
 *
 * Displays inline Alert (not full-page) to isolate failures.
 * Rest of app continues working.
 */
function ResultsErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  return (
    <Alert variant="destructive" className="my-4">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>Failed to load results</AlertTitle>
      <AlertDescription className="mt-2 space-y-2">
        <p className="text-sm">{error.message}</p>
        <Button
          variant="outline"
          size="sm"
          onClick={resetErrorBoundary}
          className="mt-2"
        >
          Retry
        </Button>
      </AlertDescription>
    </Alert>
  );
}

/**
 * Component-level error boundary for ResultCard.
 *
 * Purpose: Isolates Results component crashes - shows inline error
 * without crashing entire app.
 *
 * FE-02 & FE-03: Results crashes show inline error in results area only.
 */
export const ResultsErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary FallbackComponent={ResultsErrorFallback}>
      {children}
    </ErrorBoundary>
  );
};
