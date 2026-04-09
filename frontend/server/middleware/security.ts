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
