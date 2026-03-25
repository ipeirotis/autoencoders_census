/**
 * Tests for production error handler middleware
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import request from 'supertest';
import express, { Express, Request, Response, NextFunction } from 'express';

// Mock logger before importing errorHandler
jest.mock('../../config/logger', () => ({
  logger: {
    error: jest.fn(),
  },
}));

describe('Error handler middleware', () => {
  let app: Express;
  const originalEnv = process.env.NODE_ENV;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
  });

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
      const { logger } = await import('../../config/logger');

      await request(app).get('/test-error');

      expect(logger.error).toHaveBeenCalled();
      const logCall = (logger.error as jest.MockedFunction<any>).mock.calls[0];
      expect(logCall[0]).toContain('Request error');
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

    it('should log error with request details', async () => {
      const { logger } = await import('../../config/logger');

      await request(app).get('/test-error');

      expect(logger.error).toHaveBeenCalled();
      const logCall = (logger.error as jest.MockedFunction<any>).mock.calls[0];
      const metadata = logCall[1];

      expect(metadata.message).toBeDefined();
      expect(metadata.stack).toBeDefined();
      expect(metadata.path).toBe('/test-error');
      expect(metadata.method).toBe('GET');
    });
  });
});
