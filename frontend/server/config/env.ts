/**
 * Environment variable validation using envalid
 * Validates required environment variables at startup and provides type-safe access
 */

import { cleanEnv, str, port, makeValidator, EnvError } from 'envalid';

/**
 * Validator for the session signing secret. Enforces a minimum length so that
 * misconfigured deployments fail fast at startup instead of running with a
 * weak secret. Express-session signs cookies with this value, so a short
 * secret materially weakens session integrity.
 */
const SESSION_SECRET_MIN_LENGTH = 32;
const sessionSecret = makeValidator<string>((input) => {
  if (typeof input !== 'string' || input.length < SESSION_SECRET_MIN_LENGTH) {
    throw new EnvError(
      `SESSION_SECRET must be at least ${SESSION_SECRET_MIN_LENGTH} characters`
    );
  }
  return input;
});

/**
 * Validated environment variables
 * Server will fail fast at startup if any required variables are missing
 */
export const env = cleanEnv(process.env, {
  // Required GCP configuration
  GOOGLE_CLOUD_PROJECT: str({
    desc: 'Google Cloud project ID',
  }),
  GCS_BUCKET_NAME: str({
    desc: 'Google Cloud Storage bucket name for file uploads',
  }),

  // Required security configuration
  SESSION_SECRET: sessionSecret({
    desc: 'Secret key for session signing (min 32 characters)',
  }),
  FRONTEND_URL: str({
    desc: 'Frontend origin URL for CORS whitelist',
  }),

  // Optional configuration with defaults
  PORT: port({
    default: 5001,
    desc: 'Port for Express server',
  }),
  NODE_ENV: str({
    default: 'development',
    choices: ['development', 'test', 'production'],
    desc: 'Node environment',
  }),
});
