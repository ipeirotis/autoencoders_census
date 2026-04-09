/**
 * Production error handler middleware
 * - Hides stack traces and sensitive error details in production
 * - Logs full error details server-side
 * - Returns generic error messages in production
 */

import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import { env } from '../config/env';

/**
 * Express error handler middleware
 * MUST be registered AFTER all routes
 */
export function errorHandler(
  err: any,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Log full error details server-side (always logged regardless of environment)
  logger.error('Request error', {
    message: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
    userId: (req as any).user?.id, // If authentication is enabled
  });

  // Determine status code (preserve custom status codes)
  const status = err.status || err.statusCode || 500;

  // In production: hide error details (security)
  if (env.NODE_ENV === 'production') {
    res.status(status).json({
      error: 'Internal server error',
    });
    return;
  }

  // In development: include full error details for debugging
  res.status(status).json({
    error: err.message,
    stack: err.stack,
  });
}
