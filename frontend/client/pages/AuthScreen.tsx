/**
 * AuthScreen - the unauthenticated landing page.
 *
 * Renders a single card with a toggle between Login and Sign-up forms.
 * On success, optimistically updates the cached current user via
 * `useCurrentUser().setUser(...)` so the parent gate immediately swaps to
 * the authenticated view without waiting for a /api/auth/me round trip.
 */

import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { login, signup, AuthError } from "@/utils/auth";
import { useCurrentUser } from "@/hooks/useCurrentUser";

type Mode = "login" | "signup";

export default function AuthScreen() {
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { setUser } = useCurrentUser();

  const switchMode = (next: Mode) => {
    setMode(next);
    setError(null);
    // Keep the email across mode switches so a user who typed an email and
    // then realized they need to sign up doesn't have to retype it.
    setPassword("");
    setConfirm("");
  };

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError(null);

      if (mode === "signup" && password !== confirm) {
        setError("Passwords do not match");
        return;
      }
      if (password.length < 8) {
        setError("Password must be at least 8 characters");
        return;
      }

      setSubmitting(true);
      try {
        const user =
          mode === "login"
            ? await login(email, password)
            : await signup(email, password);
        // Optimistically populate the cached current user. The parent
        // <AuthGate> re-renders into the authenticated view immediately.
        setUser(user);
      } catch (err) {
        if (err instanceof AuthError) {
          setError(err.message);
        } else {
          setError("Unexpected error. Please try again.");
        }
      } finally {
        setSubmitting(false);
      }
    },
    [mode, email, password, confirm, setUser],
  );

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md bg-white">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-3">
            <img
              src="/AutoEncoder_logo_black.png"
              alt="Logo"
              className="w-14 h-14 object-contain"
            />
          </div>
          <CardTitle>
            {mode === "login" ? "Sign in" : "Create an account"}
          </CardTitle>
          <CardDescription>
            {mode === "login"
              ? "Sign in to upload data and run outlier analysis."
              : "Sign up to upload data and run outlier analysis."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={handleSubmit} noValidate>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={submitting}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete={
                  mode === "login" ? "current-password" : "new-password"
                }
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={submitting}
              />
              {mode === "signup" && (
                <p className="text-xs text-gray-500">
                  Must be at least 8 characters.
                </p>
              )}
            </div>

            {mode === "signup" && (
              <div className="space-y-2">
                <Label htmlFor="confirm">Confirm password</Label>
                <Input
                  id="confirm"
                  type="password"
                  autoComplete="new-password"
                  required
                  minLength={8}
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  disabled={submitting}
                />
              </div>
            )}

            {error && (
              <div
                role="alert"
                className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
              >
                {error}
              </div>
            )}

            <Button type="submit" className="w-full" disabled={submitting}>
              {submitting
                ? mode === "login"
                  ? "Signing in..."
                  : "Creating account..."
                : mode === "login"
                  ? "Sign in"
                  : "Sign up"}
            </Button>
          </form>

          <div className="mt-4 text-center text-sm text-gray-600">
            {mode === "login" ? (
              <>
                Don&apos;t have an account?{" "}
                <button
                  type="button"
                  className="text-blue-600 hover:underline font-medium"
                  onClick={() => switchMode("signup")}
                >
                  Sign up
                </button>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <button
                  type="button"
                  className="text-blue-600 hover:underline font-medium"
                  onClick={() => switchMode("login")}
                >
                  Sign in
                </button>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
