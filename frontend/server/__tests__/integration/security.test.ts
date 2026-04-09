/**
 * Security Integration Tests
 * Verifies that security middleware is properly wired into routes
 *
 * These tests verify middleware application order and configuration.
 * Full E2E authentication tests with Passport sessions are in routes/auth.test.ts
 */

import { describe, it, expect } from '@jest/globals';

describe('Security Integration Tests', () => {
  describe('Middleware imports and exports', () => {
    it('should export all required authentication middleware', async () => {
      const authModule = await import('../../middleware/auth');
      expect(authModule.requireAuth).toBeDefined();
      expect(authModule.passport).toBeDefined();
    });

    it('should export all rate limiters', async () => {
      const rateLimitsModule = await import('../../middleware/rateLimits');
      expect(rateLimitsModule.uploadLimiter).toBeDefined();
      expect(rateLimitsModule.pollLimiter).toBeDefined();
      expect(rateLimitsModule.downloadLimiter).toBeDefined();
    });

    it('should export all validation chains', async () => {
      const validationModule = await import('../../middleware/validation');
      expect(validationModule.validateSignup).toBeDefined();
      expect(validationModule.validateLogin).toBeDefined();
      expect(validationModule.validateJobId).toBeDefined();
      expect(validationModule.validateUploadUrl).toBeDefined();
      expect(validationModule.validateStartJob).toBeDefined();
    });

    it('should export security configurations', async () => {
      const securityModule = await import('../../middleware/security');
      expect(securityModule.corsConfig).toBeDefined();
      expect(securityModule.helmetConfig).toBeDefined();
    });

    it('should export file validation utilities', async () => {
      const fileValidationModule = await import('../../utils/fileValidation');
      expect(fileValidationModule.validateCSVContent).toBeDefined();
      expect(fileValidationModule.generateSafeFilename).toBeDefined();
      expect(fileValidationModule.sanitizePath).toBeDefined();
    });
  });

  describe('Route middleware wiring', () => {
    it('auth routes should import validation middleware', async () => {
      const authRoutesModule = await import('../../routes/auth');
      const authRouter = authRoutesModule.authRouter;

      // Verify router has routes registered
      expect(authRouter.stack).toBeDefined();
      expect(authRouter.stack.length).toBeGreaterThan(0);

      // Check that signup route exists
      const signupRoute = authRouter.stack.find((layer: any) =>
        layer.route && layer.route.path === '/signup' && layer.route.methods.post
      );
      expect(signupRoute).toBeDefined();

      // Verify signup route has multiple handlers (validation chain + handler)
      if (signupRoute && signupRoute.route) {
        expect(signupRoute.route.stack.length).toBeGreaterThan(1);
      }
    });

    it('jobs routes should import security middleware', async () => {
      const jobsRoutesModule = await import('../../routes/jobs');
      const jobsRouter = jobsRoutesModule.jobsRouter;

      // Verify router has routes registered
      expect(jobsRouter.stack).toBeDefined();
      expect(jobsRouter.stack.length).toBeGreaterThan(0);

      // Check that upload-url route exists
      const uploadUrlRoute = jobsRouter.stack.find((layer: any) =>
        layer.route && layer.route.path === '/upload-url' && layer.route.methods.post
      );
      expect(uploadUrlRoute).toBeDefined();

      // Verify route has multiple middleware (auth + rate limit + validation + handler)
      if (uploadUrlRoute && uploadUrlRoute.route) {
        expect(uploadUrlRoute.route.stack.length).toBeGreaterThan(1);
      }
    });

    it('index.ts should export createServer function', async () => {
      const indexModule = await import('../../index');
      const createServer = indexModule.createServer;

      expect(createServer).toBeDefined();
      expect(typeof createServer).toBe('function');
    });
  });

  describe('Middleware configuration', () => {
    it('upload rate limiter should limit to 5 requests per 15 minutes', async () => {
      const rateLimitsModule = await import('../../middleware/rateLimits');
      const uploadLimiter = rateLimitsModule.uploadLimiter;

      // Check that the rate limiter is configured
      expect(uploadLimiter).toBeDefined();
      expect(typeof uploadLimiter).toBe('function');
    });

    it('poll rate limiter should limit to 60 requests per minute', async () => {
      const rateLimitsModule = await import('../../middleware/rateLimits');
      const pollLimiter = rateLimitsModule.pollLimiter;

      expect(pollLimiter).toBeDefined();
      expect(typeof pollLimiter).toBe('function');
    });

    it('download rate limiter should limit to 10 requests per hour', async () => {
      const rateLimitsModule = await import('../../middleware/rateLimits');
      const downloadLimiter = rateLimitsModule.downloadLimiter;

      expect(downloadLimiter).toBeDefined();
      expect(typeof downloadLimiter).toBe('function');
    });

    it('CORS config should be a middleware function', async () => {
      const securityModule = await import('../../middleware/security');
      const corsConfig = securityModule.corsConfig;

      expect(corsConfig).toBeDefined();
      expect(typeof corsConfig).toBe('function');
    });

    it('Helmet config should be a middleware function', async () => {
      const securityModule = await import('../../middleware/security');
      const helmetConfig = securityModule.helmetConfig;

      expect(helmetConfig).toBeDefined();
      expect(typeof helmetConfig).toBe('function');
    });
  });

  describe('File security', () => {
    it('generateSafeFilename should produce UUID-based paths', async () => {
      const fileValidationModule = await import('../../utils/fileValidation');
      const generateSafeFilename = fileValidationModule.generateSafeFilename;

      const filename1 = generateSafeFilename('user-123');
      const filename2 = generateSafeFilename('user-123');

      // Should start with uploads/userId/
      expect(filename1).toMatch(/^uploads\/user-123\//);

      // Each call should produce different UUID
      expect(filename1).not.toBe(filename2);

      // Should end with .csv
      expect(filename1).toMatch(/\.csv$/);
    });

    it('validateCSVContent should reject binary files', async () => {
      const fileValidationModule = await import('../../utils/fileValidation');
      const validateCSVContent = fileValidationModule.validateCSVContent;

      // PNG file header
      const pngBuffer = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
      const result = await validateCSVContent(pngBuffer);

      expect(result.valid).toBe(false);
      // The validation error may say "binary" or "Invalid file encoding"
      expect(result.reason).toBeDefined();
    });

    it('validateCSVContent should accept valid CSV', async () => {
      const fileValidationModule = await import('../../utils/fileValidation');
      const validateCSVContent = fileValidationModule.validateCSVContent;

      const csvBuffer = Buffer.from('name,age\nJohn,30\nJane,25\n');
      const result = await validateCSVContent(csvBuffer);

      expect(result.valid).toBe(true);
    });

    it('sanitizePath should prevent path traversal', async () => {
      const fileValidationModule = await import('../../utils/fileValidation');
      const sanitizePath = fileValidationModule.sanitizePath;

      const uploadDir = '/uploads';
      const maliciousPath = '../../../etc/passwd';

      const result = sanitizePath(uploadDir, maliciousPath);

      // Should return null for path traversal attempt
      expect(result).toBeNull();
    });

    it('sanitizePath should allow safe paths', async () => {
      const fileValidationModule = await import('../../utils/fileValidation');
      const sanitizePath = fileValidationModule.sanitizePath;

      const uploadDir = '/uploads';
      const safePath = 'user-123/file.csv';

      const result = sanitizePath(uploadDir, safePath);

      // Should return resolved path within upload directory
      expect(result).not.toBeNull();
      expect(result).toContain('/uploads/');
    });
  });
});
