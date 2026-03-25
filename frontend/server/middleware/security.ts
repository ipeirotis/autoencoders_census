/**
 * Security middleware configuration
 * Provides CORS whitelist and helmet security headers
 */

import cors from 'cors';
import helmet from 'helmet';
import { env } from '../config/env';

// CORS whitelist - only allow requests from authorized origins
const allowedOrigins = [env.FRONTEND_URL].filter(Boolean);

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
