/**
 * Auth API client - calls /api/auth/* endpoints
 *
 * All requests use credentials: 'include' so the express-session cookie is
 * sent on cross-origin deployments (when VITE_API_BASE_URL points elsewhere).
 * The server's CORS config explicitly allows credentialed requests from
 * FRONTEND_URL.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

/**
 * Public user shape returned by the server. Mirrors `PublicUser` in
 * `frontend/server/models/user.ts`. Sensitive fields (passwordHash,
 * verificationToken, resetToken) are stripped server-side and never reach
 * the client.
 */
export interface User {
  id: string;
  email: string;
  createdAt: string;
  emailVerified: boolean;
}

/**
 * Error thrown by auth helpers. Includes the HTTP status so callers can
 * distinguish "not logged in" (401) from validation errors (400) and from
 * server errors (5xx) without re-parsing the response.
 */
export class AuthError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "AuthError";
    this.status = status;
  }
}

/**
 * Parse a server error response. The auth routes return JSON like
 * `{ error: "..." }` for ordinary failures and `{ error: "Validation
 * failed", details: [...] }` for express-validator rejections. Pick the
 * most specific message we can find.
 */
async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body?.details?.[0]?.message) return body.details[0].message;
    if (typeof body?.error === "string") return body.error;
  } catch {
    // Non-JSON response (e.g. 500 from a crash). Fall through.
  }
  return `Request failed (${res.status})`;
}

async function authFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
}

/**
 * Fetch the current authenticated user, or null if no session is active.
 * Distinguishes 401 (return null, expected for logged-out users) from other
 * failures (throw, so React Query treats them as real errors and we surface
 * them in the UI).
 */
export async function getCurrentUser(): Promise<User | null> {
  const res = await authFetch("/api/auth/me", { method: "GET" });
  if (res.status === 401) return null;
  if (!res.ok) {
    throw new AuthError(await parseError(res), res.status);
  }
  const body = await res.json();
  return body.user as User;
}

/**
 * Sign up a new account. The server logs the user in via the same response,
 * so the returned user is already authenticated for follow-up requests.
 */
export async function signup(email: string, password: string): Promise<User> {
  const res = await authFetch("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    throw new AuthError(await parseError(res), res.status);
  }
  const body = await res.json();
  return body.user as User;
}

/**
 * Log in with email + password. Server creates a session and the cookie
 * is set on the response, so subsequent calls are authenticated.
 */
export async function login(email: string, password: string): Promise<User> {
  const res = await authFetch("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    throw new AuthError(await parseError(res), res.status);
  }
  const body = await res.json();
  return body.user as User;
}

/**
 * End the current session.
 */
export async function logout(): Promise<void> {
  const res = await authFetch("/api/auth/logout", { method: "POST" });
  if (!res.ok) {
    throw new AuthError(await parseError(res), res.status);
  }
}
