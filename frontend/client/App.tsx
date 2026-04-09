import "./global.css";

import React from "react";
import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Index from "./pages/Index";
import JobProgress from "./pages/JobProgress";
import AuthScreen from "./pages/AuthScreen";
import { RootErrorBoundary } from "@/components/error-boundaries/RootErrorBoundary";
import { useCurrentUser } from "@/hooks/useCurrentUser";

const queryClient = new QueryClient();

/**
 * RequireAuth redirects unauthenticated users to "/" (which renders
 * AuthGate → AuthScreen). Used for routes whose backing API endpoints
 * are protected by requireAuth on the server.
 */
const RequireAuth = ({ children }: { children: React.ReactNode }) => {
  const { user, isLoading } = useCurrentUser();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="animate-spin w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

/**
 * AuthGate decides which top-level page to render based on session state.
 *
 * - While the initial /api/auth/me check is in flight, render a small
 *   centered spinner so the SPA does not flash either screen.
 * - If no session is active, render <AuthScreen> (login/signup forms).
 * - If a session is active, render the main <Index> upload UI.
 *
 * Background: PR #12 added requireAuth to all /api/jobs/* endpoints. Without
 * this gate, a fresh visitor would land on Index, immediately call
 * /api/jobs/upload-url, get 401, and have no UI path to recover.
 */
const AuthGate = () => {
  const { user, isLoading } = useCurrentUser();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="animate-spin w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  return user ? <Index /> : <AuthScreen />;
};

const App = () => (
  <RootErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<AuthGate />} />
            <Route path="/job/:id" element={<RequireAuth><JobProgress /></RequireAuth>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </RootErrorBoundary>
);

createRoot(document.getElementById("root")!).render(<App />);
