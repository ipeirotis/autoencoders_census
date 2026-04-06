import React, { ReactNode } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { ErrorFallback } from "./ErrorFallback";

/**
 * Full-page error boundary for App-level crashes.
 *
 * Purpose: Prevents blank screens when top-level components crash.
 * Displays centered error card with reload button.
 *
 * Recovery: window.location.reload() on reset button click.
 *
 * FE-01: App crashes display full-page error (not blank screen).
 */
export const RootErrorBoundary: React.FC<{ children: ReactNode }> = ({ children }) => {
  return (
    <ErrorBoundary
      FallbackComponent={({ error, resetErrorBoundary }) => (
        <div className="min-h-screen flex items-center justify-center bg-slate-50 p-4">
          <ErrorFallback error={error} resetErrorBoundary={resetErrorBoundary} />
        </div>
      )}
      onReset={() => {
        window.location.reload();
      }}
    >
      {children}
    </ErrorBoundary>
  );
};
