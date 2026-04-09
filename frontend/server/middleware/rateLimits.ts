/**
 * Rate limiting middleware
 * Prevents abuse of upload, polling, and download endpoints
 *
 * Uses express-rate-limit 8.0.2+ which fixes CVE-2026-30827
 * (IPv4-mapped IPv6 address bypass vulnerability)
 */

import rateLimit, { ipKeyGenerator } from 'express-rate-limit';
import type { Request } from 'express';

/**
 * Key generator for rate limiting
 * Uses user ID if authenticated, otherwise falls back to IP with proper IPv6 handling
 */
const userOrIpKeyGenerator = (req: Request): string => {
  return (req as any).user?.id || ipKeyGenerator(req);
};

/**
 * Upload rate limiter
 * Limits: 5 requests per 15 minutes per user
 *
 * Applied to endpoints that actually kick off a training job
 * (/api/jobs/start-job and the /api/upload fallback). Prevents Vertex AI
 * quota abuse. A separate, independent bucket (`uploadUrlLimiter`) is used
 * for the cheap /upload-url step so that a normal 3-step upload flow
 * consumes one slot from each bucket rather than halving the apparent
 * "uploads per 15 min" budget.
 */
export const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5,
  keyGenerator: userOrIpKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Upload limit exceeded. Try again in 15 minutes.' },
});

/**
 * Signed-URL request rate limiter
 * Limits: 5 requests per 15 minutes per user
 *
 * Applied to /api/jobs/upload-url. Kept as an independent bucket from
 * `uploadLimiter` so a normal 3-step upload (one /upload-url call followed
 * by one /start-job call) does not consume two slots from the same budget.
 */
export const uploadUrlLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5,
  keyGenerator: userOrIpKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many upload URL requests. Try again in 15 minutes.' },
});

/**
 * Authentication rate limiter
 * Limits: 10 requests per 15 minutes per IP
 *
 * Applied to login, signup, and password reset request/reset endpoints.
 * Throttles online brute-force and credential-stuffing attacks while still
 * accommodating legitimate users who mistype a password a few times. Keyed
 * by IP only (not user ID) since unauthenticated requests have no user.
 */
export const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10,
  keyGenerator: (req: Request): string => ipKeyGenerator(req),
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many authentication attempts. Try again in 15 minutes.' },
});

/**
 * Poll rate limiter
 * Limits: 60 requests per minute per user
 * Prevents excessive status checking
 */
export const pollLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60,
  keyGenerator: userOrIpKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many status checks. Slow down polling.' },
});

/**
 * Download rate limiter
 * Limits: 10 requests per hour per user
 * Prevents excessive result downloads
 */
export const downloadLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 10,
  keyGenerator: userOrIpKeyGenerator,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Download limit exceeded. Try again in an hour.' },
});
