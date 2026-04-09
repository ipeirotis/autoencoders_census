/**
 * Tests for express-validator input validation middleware
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import request from 'supertest';
import express, { Express } from 'express';
import {
  validateSignup,
  validateLogin,
  validateJobId,
  validateUploadUrl,
  validateStartJob,
  handleValidationErrors,
} from '../../middleware/validation';

describe('Auth validation middleware', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(express.json());
  });

  describe('validateSignup', () => {
    beforeEach(() => {
      app.post('/test-signup', validateSignup, (req, res) => {
        res.json({ success: true });
      });
    });

    it('should reject invalid email format', async () => {
      const response = await request(app)
        .post('/test-signup')
        .send({ email: 'not-an-email', password: 'password123' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'email',
            message: expect.stringContaining('email'),
          }),
        ])
      );
    });

    it('should reject password shorter than 8 chars', async () => {
      const response = await request(app)
        .post('/test-signup')
        .send({ email: 'test@example.com', password: 'short' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'password',
            message: expect.stringContaining('8'),
          }),
        ])
      );
    });

    it('should accept valid email and password', async () => {
      const response = await request(app)
        .post('/test-signup')
        .send({ email: 'test@example.com', password: 'password123' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should normalize email (lowercase)', async () => {
      const response = await request(app)
        .post('/test-signup')
        .send({ email: 'TEST@EXAMPLE.COM', password: 'password123' });

      expect(response.status).toBe(200);
    });
  });

  describe('validateLogin', () => {
    beforeEach(() => {
      app.post('/test-login', validateLogin, (req, res) => {
        res.json({ success: true });
      });
    });

    it('should reject missing email field', async () => {
      const response = await request(app)
        .post('/test-login')
        .send({ password: 'password123' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'email',
          }),
        ])
      );
    });

    it('should reject missing password field', async () => {
      const response = await request(app)
        .post('/test-login')
        .send({ email: 'test@example.com' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'password',
          }),
        ])
      );
    });

    it('should accept valid credentials', async () => {
      const response = await request(app)
        .post('/test-login')
        .send({ email: 'test@example.com', password: 'password123' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });

  describe('handleValidationErrors', () => {
    it('should return 400 with field-level errors', async () => {
      app.post('/test-validation', validateSignup, (req, res) => {
        res.json({ success: true });
      });

      const response = await request(app)
        .post('/test-validation')
        .send({ email: 'invalid', password: 'short' });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body).toHaveProperty('details');
      expect(Array.isArray(response.body.details)).toBe(true);
      expect(response.body.details.length).toBeGreaterThan(0);
      expect(response.body.details[0]).toHaveProperty('field');
      expect(response.body.details[0]).toHaveProperty('message');
    });
  });
});

describe('Jobs validation middleware', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(express.json());
  });

  describe('validateJobId', () => {
    beforeEach(() => {
      app.get('/test-job/:id', validateJobId, (req, res) => {
        res.json({ success: true });
      });
    });

    it('should reject non-UUID job ID', async () => {
      const response = await request(app).get('/test-job/not-a-uuid');

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'id',
            message: expect.stringContaining('Invalid job ID'),
          }),
        ])
      );
    });

    it('should accept valid UUID', async () => {
      const validUuid = '123e4567-e89b-12d3-a456-426614174000';
      const response = await request(app).get(`/test-job/${validUuid}`);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });

  describe('validateUploadUrl', () => {
    beforeEach(() => {
      app.post('/test-upload-url', validateUploadUrl, (req, res) => {
        res.json({ success: true });
      });
    });

    it('should reject missing filename', async () => {
      const response = await request(app)
        .post('/test-upload-url')
        .send({ contentType: 'text/csv' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'filename',
            message: expect.stringContaining('required'),
          }),
        ])
      );
    });

    it('should reject non-CSV filename', async () => {
      const response = await request(app)
        .post('/test-upload-url')
        .send({ filename: 'data.txt', contentType: 'text/plain' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'filename',
            message: expect.stringContaining('CSV'),
          }),
        ])
      );
    });

    it('should accept CSV filename', async () => {
      const response = await request(app)
        .post('/test-upload-url')
        .send({ filename: 'data.csv', contentType: 'text/csv' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should accept CSV filename (case insensitive)', async () => {
      const response = await request(app)
        .post('/test-upload-url')
        .send({ filename: 'data.CSV', contentType: 'text/csv' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });

  describe('validateStartJob', () => {
    beforeEach(() => {
      app.post('/test-start-job', validateStartJob, (req, res) => {
        res.json({ success: true });
      });
    });

    it('should reject missing jobId', async () => {
      const response = await request(app)
        .post('/test-start-job')
        .send({ gcsFileName: 'file.csv' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'jobId',
          }),
        ])
      );
    });

    it('should reject missing gcsFileName', async () => {
      const validUuid = '123e4567-e89b-12d3-a456-426614174000';
      const response = await request(app)
        .post('/test-start-job')
        .send({ jobId: validUuid });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
      expect(response.body.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: 'gcsFileName',
            message: expect.stringContaining('required'),
          }),
        ])
      );
    });

    it('should accept valid jobId and gcsFileName', async () => {
      const validUuid = '123e4567-e89b-12d3-a456-426614174000';
      const response = await request(app)
        .post('/test-start-job')
        .send({ jobId: validUuid, gcsFileName: 'uploads/user123/file.csv' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });
});
