/**
 * Security middleware configuration
 * Provides CORS whitelist and helmet security headers
 */

import cors from 'cors';
import helmet from 'helmet';
import { env } from '../config/env';

/**
 * CORS whitelist - only allow requests from authorized origins.
 *
 * Browsers send the `Origin` header as a bare scheme+host+port (e.g.
 * `https://app.example.com`) with no trailing slash and no path. If the
 * configured FRONTEND_URL has a trailing slash or includes a path, exact
 * string matching against the raw env value will never succeed and every
 * credentialed request gets blocked. Normalize via `new URL(...).origin` so
 * easy deployment-time misconfigurations don't lock the SPA out.
 *
 * If FRONTEND_URL is not a parseable URL we fall back to the raw value and
 * log a warning rather than crashing the server, so the misconfiguration is
 * visible at boot but does not take the API down hard.
 */
function normalizeOrigin(value: string): string {
  try {
    return new URL(value).origin;
  } catch {
    // eslint-disable-next-line no-console
    console.warn(
      `[cors] FRONTEND_URL=${value} is not a valid URL; CORS allowlist will use the raw value`,
    );
    return value;
  }
}

const allowedOrigins = [normalizeOrigin(env.FRONTEND_URL)].filter(Boolean);

/**
 * CSRF defense via Origin-header verification.
 *
 * With `sameSite: 'none'` the session cookie rides cross-site requests, so
 * a third-party HTML form can fire mutating POSTs (logout, start-job, etc.)
 * with the victim's session attached. CORS blocks _reading_ the response
 * but not the side effect.
 *
 * Fix: on mutating methods (POST/PUT/PATCH/DELETE), verify the Origin
 * header against the allowlist. Browsers always send Origin on cross-origin
 * requests, so a forged form POST from evil.com will carry
 * `Origin: https://evil.com` and be rejected. Same-origin requests may
 * omit Origin entirely — those are allowed because they can't be spoofed
 * by a cross-site form. Non-browser clients (curl, server-to-server) also
 * omit Origin and are safe because they don't carry browser cookies.
 */
import { Request, Response, NextFunction } from 'express';

const MUTATING_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

export function csrfOriginCheck(req: Request, res: Response, next: NextFunction): void {
  if (!MUTATING_METHODS.has(req.method)) {
    next();
    return;
  }

  const origin = req.get('origin');

  // No Origin header = same-origin browser request or non-browser client.
  // Both are safe to allow.
  if (!origin) {
    next();
    return;
  }

  // Cross-origin: verify the origin is in the allowlist.
  if (allowedOrigins.includes(origin)) {
    next();
    return;
  }

  res.status(403).json({ error: 'Origin not allowed' });
}

/**
 * CORS configuration with origin whitelist
 * - Allows requests from whitelisted origins only
 * - Allows requests with no origin (mobile apps, curl)
 * - Enables credentials for session-based auth
 */
export const corsConfig = cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, curl, server-to-server)
    if (!origin) {
      callback(null, true);
      return;
    }

    // Check if origin is in whitelist
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('CORS policy violation'));
    }
  },
  credentials: true, // Allow cookies for session auth
});

/**
 * Helmet configuration for security headers
 * - X-Frame-Options: DENY (prevent clickjacking)
 * - X-Content-Type-Options: nosniff (prevent MIME sniffing)
 * - Content-Security-Policy with Vite-compatible settings
 */
export const helmetConfig = helmet({
  frameguard: {
    action: 'deny', // X-Frame-Options: DENY
  },
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"], // 'unsafe-inline' needed for Vite dev HMR
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'", env.FRONTEND_URL].filter(Boolean),
    },
  },
});
