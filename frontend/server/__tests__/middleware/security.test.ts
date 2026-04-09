/**
 * Security middleware tests
 * Tests CORS whitelist and helmet security headers
 */

import request from 'supertest';
import express, { Express } from 'express';
import { corsConfig, helmetConfig } from '../../middleware/security';

describe('CORS Configuration', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(corsConfig);
    app.get('/test', (_req, res) => res.json({ message: 'test' }));
  });

  it('should allow requests from FRONTEND_URL origin', async () => {
    const response = await request(app)
      .get('/test')
      .set('Origin', 'http://localhost:5173');

    expect(response.status).toBe(200);
    expect(response.headers['access-control-allow-origin']).toBe('http://localhost:5173');
  });

  it('should reject requests from unauthorized origins', async () => {
    const response = await request(app)
      .get('/test')
      .set('Origin', 'https://evil.com');

    // CORS error should block the request
    expect(response.status).toBe(500);
  });

  it('should allow requests with no origin header (mobile apps, curl)', async () => {
    const response = await request(app)
      .get('/test');

    expect(response.status).toBe(200);
  });

  it('should enable credentials (Access-Control-Allow-Credentials: true)', async () => {
    const response = await request(app)
      .get('/test')
      .set('Origin', 'http://localhost:5173');

    expect(response.headers['access-control-allow-credentials']).toBe('true');
  });

  it('should handle CORS preflight OPTIONS requests', async () => {
    const response = await request(app)
      .options('/test')
      .set('Origin', 'http://localhost:5173')
      .set('Access-Control-Request-Method', 'GET');

    expect(response.status).toBe(204);
    expect(response.headers['access-control-allow-origin']).toBe('http://localhost:5173');
  });
});

describe('Helmet Security Headers', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(helmetConfig);
    app.get('/test', (_req, res) => res.json({ message: 'test' }));
  });

  it('should include X-Frame-Options: DENY header', async () => {
    const response = await request(app).get('/test');

    expect(response.headers['x-frame-options']).toBe('DENY');
  });

  it('should include X-Content-Type-Options: nosniff header', async () => {
    const response = await request(app).get('/test');

    expect(response.headers['x-content-type-options']).toBe('nosniff');
  });

  it('should include Content-Security-Policy header', async () => {
    const response = await request(app).get('/test');

    expect(response.headers['content-security-policy']).toBeDefined();
  });

  it('should allow scripts from self and unsafe-inline (for Vite)', async () => {
    const response = await request(app).get('/test');

    const csp = response.headers['content-security-policy'];
    expect(csp).toContain("script-src 'self' 'unsafe-inline'");
  });

  it('should allow connections to self and FRONTEND_URL', async () => {
    const response = await request(app).get('/test');

    const csp = response.headers['content-security-policy'];
    expect(csp).toContain('connect-src');
    expect(csp).toContain("'self'");
  });
});
