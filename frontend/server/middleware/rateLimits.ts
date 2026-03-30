/**
 * Rate limiting middleware
 * Prevents abuse of upload, polling, and download endpoints
 *
 * Uses express-rate-limit 8.0.2+ which fixes CVE-2026-30827
 * (IPv4-mapped IPv6 address bypass vulnerability)
 */

import rateLimit, { ipKeyGenerator } from 'express-rate-limit';

/**
 * Upload rate limiter
 * Limits: 5 requests per 15 minutes per user
 * Prevents Vertex AI quota abuse
 */
export const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5,
  keyGenerator: (req) => {
    // Use user ID if authenticated, otherwise fallback to IP with proper IPv6 handling
    return (req as any).user?.id || ipKeyGenerator(req);
  },
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Upload limit exceeded. Try again in 15 minutes.' },
});

/**
 * Poll rate limiter
 * Limits: 60 requests per minute per user
 * Prevents excessive status checking
 */
export const pollLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60,
  keyGenerator: (req) => {
    // Use user ID if authenticated, otherwise fallback to IP with proper IPv6 handling
    return (req as any).user?.id || ipKeyGenerator(req);
  },
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
  keyGenerator: (req) => {
    // Use user ID if authenticated, otherwise fallback to IP with proper IPv6 handling
    return (req as any).user?.id || ipKeyGenerator(req);
  },
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Download limit exceeded. Try again in an hour.' },
});
