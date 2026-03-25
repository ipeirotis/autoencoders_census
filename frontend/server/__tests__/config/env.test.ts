/**
 * Tests for environment variable validation
 *
 * Note: envalid validates environment variables at module import time.
 * These tests verify that the env module exports the expected values
 * given the environment variables set in setup.ts.
 */

import { describe, it, expect } from '@jest/globals';
import { env } from '../../config/env';

describe('Environment validation', () => {
  // Environment variables are set in __tests__/setup.ts before tests run
  // envalid validates them when the module is imported

  it('should export validated env object', () => {
    expect(env).toBeDefined();
    expect(typeof env).toBe('object');
  });

  it('should include required GOOGLE_CLOUD_PROJECT', () => {
    expect(env.GOOGLE_CLOUD_PROJECT).toBe('test-project');
  });

  it('should include required GCS_BUCKET_NAME', () => {
    expect(env.GCS_BUCKET_NAME).toBe('test-bucket');
  });

  it('should include required SESSION_SECRET', () => {
    expect(env.SESSION_SECRET).toBe('test-secret-key-minimum-32-characters-long');
  });

  it('should include required FRONTEND_URL', () => {
    expect(env.FRONTEND_URL).toBe('http://localhost:5173');
  });

  it('should default PORT to 5001 when not specified', () => {
    expect(env.PORT).toBe(5001);
  });

  it('should use NODE_ENV from setup (test)', () => {
    expect(env.NODE_ENV).toBe('test');
  });

  it('should make env properties readonly (envalid freezes object)', () => {
    expect(() => {
      // @ts-expect-error - Testing runtime behavior
      env.PORT = 9999;
    }).toThrow();
  });
});
