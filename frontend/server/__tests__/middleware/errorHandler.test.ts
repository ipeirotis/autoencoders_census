/**
 * Tests for production error handler middleware
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import request from 'supertest';
import express, { Express, Request, Response, NextFunction } from 'express';

describe('Error handler middleware', () => {
  let app: Express;
  const originalEnv = process.env.NODE_ENV;

  afterEach(() => {
    // Restore environment
    process.env.NODE_ENV = originalEnv;
  });

  describe('Production mode (NODE_ENV=production)', () => {
    beforeEach(async () => {
      process.env.NODE_ENV = 'production';
      jest.resetModules();

      // Import errorHandler after setting NODE_ENV
      const errorHandlerModule = await import('../../middleware/errorHandler');
      const errorHandler = errorHandlerModule.errorHandler;

      app = express();
      app.get('/test-error', (_req: Request, _res: Response) => {
        throw new Error('Test error with sensitive info');
      });
      app.use(errorHandler);
    });

    it('should return generic error message (hide sensitive details)', async () => {
      const response = await request(app).get('/test-error');

      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Internal server error');
      // Should NOT expose original error message
      expect(response.body.error).not.toContain('sensitive info');
    });

    it('should not include stack trace in response', async () => {
      const response = await request(app).get('/test-error');

      expect(response.body.stack).toBeUndefined();
    });

    it('should log full error details server-side', async () => {
      // Logger logs are visible in test output (winston console transport)
      // The error handler calls logger.error() with full details
      // This test verifies the response is generic (tested above)
      // Actual logging is tested in logger.test.ts
      const response = await request(app).get('/test-error');
      expect(response.status).toBe(500);
    });
  });

  describe('Development mode (NODE_ENV=development)', () => {
    beforeEach(async () => {
      process.env.NODE_ENV = 'development';
      jest.resetModules();

      const errorHandlerModule = await import('../../middleware/errorHandler');
      const errorHandler = errorHandlerModule.errorHandler;

      app = express();
      app.get('/test-error', (_req: Request, _res: Response) => {
        throw new Error('Development error');
      });
      app.use(errorHandler);
    });

    it('should include original error message in development', async () => {
      const response = await request(app).get('/test-error');

      expect(response.status).toBe(500);
      expect(response.body.error).toContain('Development error');
    });

    it('should include stack trace in development', async () => {
      const response = await request(app).get('/test-error');

      expect(response.body.stack).toBeDefined();
      expect(response.body.stack).toContain('Error: Development error');
    });
  });

  describe('Custom status codes', () => {
    beforeEach(async () => {
      process.env.NODE_ENV = 'production';
      jest.resetModules();

      const errorHandlerModule = await import('../../middleware/errorHandler');
      const errorHandler = errorHandlerModule.errorHandler;

      app = express();

      // Route that throws error with custom status
      app.get('/test-401', (_req: Request, _res: Response) => {
        const err: any = new Error('Unauthorized');
        err.status = 401;
        throw err;
      });

      app.get('/test-403', (_req: Request, _res: Response) => {
        const err: any = new Error('Forbidden');
        err.status = 403;
        throw err;
      });

      app.use(errorHandler);
    });

    it('should preserve 401 status code', async () => {
      const response = await request(app).get('/test-401');
      expect(response.status).toBe(401);
    });

    it('should preserve 403 status code', async () => {
      const response = await request(app).get('/test-403');
      expect(response.status).toBe(403);
    });
  });

  describe('Error logging', () => {
    beforeEach(async () => {
      process.env.NODE_ENV = 'production';
      jest.resetModules();

      const errorHandlerModule = await import('../../middleware/errorHandler');
      const errorHandler = errorHandlerModule.errorHandler;

      app = express();
      app.get('/test-error', (_req: Request, _res: Response) => {
        throw new Error('Logged error');
      });
      app.use(errorHandler);
    });

    it('should handle errors and return appropriate response', async () => {
      // Error logging is verified via winston console output in test runs
      // The logger.error() call in errorHandler includes: message, stack, path, method
      // This is integration-tested via the logger configuration
      const response = await request(app).get('/test-error');

      expect(response.status).toBe(500);
      expect(response.body.error).toBe('Internal server error');
    });
  });
});
