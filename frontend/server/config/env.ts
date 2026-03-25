/**
 * Environment configuration
 * Centralizes environment variable access with type safety
 */

export const env = {
  // Frontend origin for CORS whitelist
  FRONTEND_URL: process.env.FRONTEND_URL || 'http://localhost:5173',

  // Google Cloud config
  GOOGLE_CLOUD_PROJECT: process.env.GOOGLE_CLOUD_PROJECT || 'your-project-id',
  GCS_BUCKET_NAME: process.env.GCS_BUCKET_NAME || 'your-bucket-name',

  // Server config
  PORT: parseInt(process.env.PORT || '5001', 10),
  NODE_ENV: process.env.NODE_ENV || 'development',
};
