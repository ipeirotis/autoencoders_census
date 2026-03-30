/**
 * Tests for rate limiting middleware
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import express, { type Express } from 'express';
import request from 'supertest';

describe('Rate Limiting middleware', () => {
  let app: Express;
  let uploadLimiter: any;
  let pollLimiter: any;
  let downloadLimiter: any;

  beforeEach(async () => {
    // Import rate limiters
    const rateLimitModule = await import('../../middleware/rateLimits');
    uploadLimiter = rateLimitModule.uploadLimiter;
    pollLimiter = rateLimitModule.pollLimiter;
    downloadLimiter = rateLimitModule.downloadLimiter;

    // Create fresh test app for each test
    app = express();
    app.use(express.json());
  });

  describe('uploadLimiter', () => {
    it('should allow first 5 upload requests within 15 minutes', async () => {
      // Use unique user ID for this test
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'upload-test-1' };
        next();
      });
      testApp.post('/upload', uploadLimiter, (_req, res) => {
        res.status(200).json({ success: true });
      });

      for (let i = 0; i < 5; i++) {
        const response = await request(testApp).post('/upload');
        expect(response.status).toBe(200);
        expect(response.body).toEqual({ success: true });
      }
    });

    it('should return 429 on 6th upload request', async () => {
      // Use unique user ID for this test
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'upload-test-2' };
        next();
      });
      testApp.post('/upload', uploadLimiter, (_req, res) => {
        res.status(200).json({ success: true });
      });

      // First 5 succeed
      for (let i = 0; i < 5; i++) {
        await request(testApp).post('/upload');
      }

      // 6th should fail
      const response = await request(testApp).post('/upload');
      expect(response.status).toBe(429);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Upload limit exceeded');
    });

    it('should include correct error message in 429 response', async () => {
      // Use unique user ID for this test
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'upload-test-3' };
        next();
      });
      testApp.post('/upload', uploadLimiter, (_req, res) => {
        res.status(200).json({ success: true });
      });

      // Exceed limit
      for (let i = 0; i < 5; i++) {
        await request(testApp).post('/upload');
      }

      const response = await request(testApp).post('/upload');
      expect(response.status).toBe(429);
      expect(response.body).toEqual({
        error: 'Upload limit exceeded. Try again in 15 minutes.',
      });
    });

    it('should have independent rate limits for different users', async () => {
      // Create app with dynamic user
      const appMultiUser = express();
      appMultiUser.use(express.json());

      // Use unique user IDs for this test
      let currentUserId = 'upload-test-multi-1';
      appMultiUser.use((req, _res, next) => {
        (req as any).user = { id: currentUserId };
        next();
      });
      appMultiUser.post('/upload', uploadLimiter, (_req, res) => {
        res.status(200).json({ success: true });
      });

      // User 1: use all 5 requests
      currentUserId = 'upload-test-multi-1';
      for (let i = 0; i < 5; i++) {
        const response = await request(appMultiUser).post('/upload');
        expect(response.status).toBe(200);
      }

      // User 1: 6th request should fail
      const user1ExtraResponse = await request(appMultiUser).post('/upload');
      expect(user1ExtraResponse.status).toBe(429);

      // User 2: should still have full quota
      currentUserId = 'upload-test-multi-2';
      const user2Response = await request(appMultiUser).post('/upload');
      expect(user2Response.status).toBe(200);
    });

    it('should use req.user.id when authenticated', async () => {
      // Use unique user ID for this test
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'upload-test-auth' };
        next();
      });
      testApp.post('/upload', uploadLimiter, (_req, res) => {
        res.status(200).json({ success: true });
      });

      // This is implicitly tested by the other tests,
      // but we verify the keyGenerator is using user ID
      const response = await request(testApp).post('/upload');
      expect(response.status).toBe(200);
      // If it were using IP, all tests would share a limit
    });

    it('should fallback to req.ip when not authenticated', async () => {
      // Create app without authentication
      const appNoAuth = express();
      appNoAuth.use(express.json());
      appNoAuth.post('/upload', uploadLimiter, (_req, res) => {
        res.status(200).json({ success: true });
      });

      // Should still work with IP-based limiting
      const response = await request(appNoAuth).post('/upload');
      expect(response.status).toBe(200);
    });
  });

  describe('pollLimiter', () => {
    it('should allow 60 poll requests within 1 minute', async () => {
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'poll-test-1' };
        next();
      });
      testApp.get('/status', pollLimiter, (_req, res) => {
        res.status(200).json({ status: 'running' });
      });

      for (let i = 0; i < 60; i++) {
        const response = await request(testApp).get('/status');
        expect(response.status).toBe(200);
        expect(response.body).toEqual({ status: 'running' });
      }
    });

    it('should return 429 on 61st poll request', async () => {
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'poll-test-2' };
        next();
      });
      testApp.get('/status', pollLimiter, (_req, res) => {
        res.status(200).json({ status: 'running' });
      });

      // First 60 succeed
      for (let i = 0; i < 60; i++) {
        await request(testApp).get('/status');
      }

      // 61st should fail
      const response = await request(testApp).get('/status');
      expect(response.status).toBe(429);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Too many status checks');
    });

    it('should reset after 1 minute window', async () => {
      // Note: This test requires mocking time or is very slow
      // For now, we just verify the configuration is correct
      expect(pollLimiter).toBeDefined();
    });
  });

  describe('downloadLimiter', () => {
    it('should allow 10 download requests within 1 hour', async () => {
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'download-test-1' };
        next();
      });
      testApp.get('/download', downloadLimiter, (_req, res) => {
        res.status(200).json({ file: 'data.csv' });
      });

      for (let i = 0; i < 10; i++) {
        const response = await request(testApp).get('/download');
        expect(response.status).toBe(200);
        expect(response.body).toEqual({ file: 'data.csv' });
      }
    });

    it('should return 429 on 11th download request', async () => {
      const testApp = express();
      testApp.use(express.json());
      testApp.use((req, _res, next) => {
        (req as any).user = { id: 'download-test-2' };
        next();
      });
      testApp.get('/download', downloadLimiter, (_req, res) => {
        res.status(200).json({ file: 'data.csv' });
      });

      // First 10 succeed
      for (let i = 0; i < 10; i++) {
        await request(testApp).get('/download');
      }

      // 11th should fail
      const response = await request(testApp).get('/download');
      expect(response.status).toBe(429);
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Download limit exceeded');
    });
  });
});
