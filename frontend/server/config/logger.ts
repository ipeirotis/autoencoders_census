/**
 * Winston logger with Cloud Logging integration
 * - Logs to Cloud Logging in production
 * - Logs to console in development/test
 */

import winston from 'winston';
import { LoggingWinston } from '@google-cloud/logging-winston';
import { env } from './env';

const transports: winston.transport[] = [];

// In production, use Cloud Logging transport
if (env.NODE_ENV === 'production') {
  const loggingWinston = new LoggingWinston({
    projectId: env.GOOGLE_CLOUD_PROJECT,
    // Log name will appear in Cloud Logging
    logName: 'autoencoder-server',
  });
  transports.push(loggingWinston);
}

// In development/test, use console transport
if (env.NODE_ENV !== 'production') {
  transports.push(
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      ),
    })
  );
}

/**
 * Winston logger instance
 * - Logs to Cloud Logging in production
 * - Logs to console in development/test
 * - Supports structured logging with metadata
 */
export const logger = winston.createLogger({
  level: env.NODE_ENV === 'production' ? 'info' : 'debug',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports,
  // Don't exit on handled exceptions
  exitOnError: false,
});
