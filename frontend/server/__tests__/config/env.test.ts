/**
 * Tests for environment variable validation
 */

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';

describe('Environment validation', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    // Reset environment before each test
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  it('should throw when GOOGLE_CLOUD_PROJECT is missing', async () => {
    delete process.env.GOOGLE_CLOUD_PROJECT;
    process.env.GCS_BUCKET_NAME = 'test-bucket';
    process.env.SESSION_SECRET = 'test-secret-at-least-32-chars-long';
    process.env.FRONTEND_URL = 'http://localhost:5173';

    await expect(async () => {
      await import('../../config/env.js');
    }).rejects.toThrow();
  });

  it('should throw when GCS_BUCKET_NAME is missing', async () => {
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
    delete process.env.GCS_BUCKET_NAME;
    process.env.SESSION_SECRET = 'test-secret-at-least-32-chars-long';
    process.env.FRONTEND_URL = 'http://localhost:5173';

    await expect(async () => {
      await import('../../config/env.js');
    }).rejects.toThrow();
  });

  it('should throw when SESSION_SECRET is missing', async () => {
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
    process.env.GCS_BUCKET_NAME = 'test-bucket';
    delete process.env.SESSION_SECRET;
    process.env.FRONTEND_URL = 'http://localhost:5173';

    await expect(async () => {
      await import('../../config/env.js');
    }).rejects.toThrow();
  });

  it('should pass when all required variables are present', async () => {
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
    process.env.GCS_BUCKET_NAME = 'test-bucket';
    process.env.SESSION_SECRET = 'test-secret-at-least-32-chars-long';
    process.env.FRONTEND_URL = 'http://localhost:5173';

    await expect(async () => {
      const envModule = await import('../../config/env.js');
      expect(envModule.env).toBeDefined();
    }).resolves.not.toThrow();
  });

  it('should default PORT to 5001 when not specified', async () => {
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
    process.env.GCS_BUCKET_NAME = 'test-bucket';
    process.env.SESSION_SECRET = 'test-secret-at-least-32-chars-long';
    process.env.FRONTEND_URL = 'http://localhost:5173';
    delete process.env.PORT;

    const envModule = await import('../../config/env.js');
    expect(envModule.env.PORT).toBe(5001);
  });

  it('should default NODE_ENV to development when not specified', async () => {
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
    process.env.GCS_BUCKET_NAME = 'test-bucket';
    process.env.SESSION_SECRET = 'test-secret-at-least-32-chars-long';
    process.env.FRONTEND_URL = 'http://localhost:5173';
    delete process.env.NODE_ENV;

    const envModule = await import('../../config/env.js');
    expect(envModule.env.NODE_ENV).toBe('development');
  });
});
