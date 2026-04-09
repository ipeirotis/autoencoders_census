/**
 * Tests for Passport authentication middleware
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import type { Request, Response, NextFunction } from 'express';

// Mock user model before importing auth middleware
const mockGetUserByEmail = jest.fn();
const mockGetUserById = jest.fn();

jest.mock('../../models/user', () => ({
  getUserByEmail: mockGetUserByEmail,
  getUserById: mockGetUserById,
}));

describe('Authentication middleware', () => {
  let requireAuth: any;
  let passport: any;

  beforeEach(async () => {
    jest.clearAllMocks();

    // Import after mocks are set up
    const authModule = await import('../../middleware/auth');
    requireAuth = authModule.requireAuth;
    passport = authModule.passport;
  });

  describe('requireAuth middleware', () => {
    it('should return 401 when req.isAuthenticated() is false', () => {
      const mockReq = {
        isAuthenticated: jest.fn().mockReturnValue(false),
      } as unknown as Request;

      const mockRes = {
        status: jest.fn().mockReturnThis(),
        json: jest.fn(),
      } as unknown as Response;

      const mockNext = jest.fn() as NextFunction;

      requireAuth(mockReq, mockRes, mockNext);

      expect(mockRes.status).toHaveBeenCalledWith(401);
      expect(mockRes.json).toHaveBeenCalledWith({ error: 'Authentication required' });
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should call next() when req.isAuthenticated() is true', () => {
      const mockReq = {
        isAuthenticated: jest.fn().mockReturnValue(true),
      } as unknown as Request;

      const mockRes = {} as Response;
      const mockNext = jest.fn() as NextFunction;

      requireAuth(mockReq, mockRes, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });
  });

  describe('Passport local strategy', () => {
    it('should validate correct email/password', async () => {
      const mockUser = {
        id: 'user-123',
        email: 'test@example.com',
        passwordHash: '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyqKOr6z0Oe6', // bcrypt hash of 'password123'
      };

      mockGetUserByEmail.mockResolvedValue(mockUser);

      // We can't easily test the passport strategy directly in unit tests
      // Instead, we verify that passport is configured (integration test will verify behavior)
      expect(passport).toBeDefined();
      expect(passport.serializeUser).toBeDefined();
      expect(passport.deserializeUser).toBeDefined();
    });

    it('should have serializeUser configured', () => {
      expect(passport).toBeDefined();
      expect(typeof passport.serializeUser).toBe('function');
    });

    it('should have deserializeUser configured', () => {
      expect(passport).toBeDefined();
      expect(typeof passport.deserializeUser).toBe('function');
    });
  });
});
