/**
 * Authentication routes
 * Handles user signup, login, logout, and current user retrieval
 */

import { Router, Request, Response } from 'express';
import { passport, requireAuth } from '../middleware/auth';
import { createUser } from '../models/user';
import { logger } from '../config/logger';

const router = Router();

/**
 * POST /api/auth/signup
 * Create a new user account
 */
router.post('/signup', async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    if (password.length < 8) {
      return res.status(400).json({ error: 'Password must be at least 8 characters' });
    }

    // Create user
    const user = await createUser(email, password);

    // Log the user in
    req.login(user, (err) => {
      if (err) {
        logger.error('Login after signup failed', { error: err });
        return res.status(500).json({ error: 'Failed to log in after signup' });
      }

      logger.info('User signed up and logged in', { userId: user.id, email: user.email });
      res.status(201).json({ user });
    });
  } catch (error) {
    if (error instanceof Error && error.message === 'Email already registered') {
      return res.status(400).json({ error: error.message });
    }

    logger.error('Signup failed', { error: error instanceof Error ? error.message : String(error) });
    res.status(500).json({ error: 'Failed to create account' });
  }
});

/**
 * POST /api/auth/login
 * Authenticate user with email and password
 */
router.post('/login', (req: Request, res: Response, next) => {
  passport.authenticate('local', (err: any, user: any, info: any) => {
    if (err) {
      logger.error('Login error', { error: err });
      return res.status(500).json({ error: 'Login failed' });
    }

    if (!user) {
      return res.status(401).json({ error: info?.message || 'Invalid credentials' });
    }

    req.login(user, (loginErr) => {
      if (loginErr) {
        logger.error('Session creation failed', { error: loginErr });
        return res.status(500).json({ error: 'Failed to create session' });
      }

      logger.info('User logged in', { userId: user.id, email: user.email });

      // Don't send passwordHash to client
      const { passwordHash, ...userPublic } = user;
      res.json({ user: userPublic });
    });
  })(req, res, next);
});

/**
 * POST /api/auth/logout
 * End user session
 */
router.post('/logout', (req: Request, res: Response) => {
  const userId = (req.user as any)?.id;

  req.logout((err) => {
    if (err) {
      logger.error('Logout failed', { error: err, userId });
      return res.status(500).json({ error: 'Failed to log out' });
    }

    logger.info('User logged out', { userId });
    res.json({ message: 'Logged out successfully' });
  });
});

/**
 * GET /api/auth/me
 * Get current authenticated user
 */
router.get('/me', requireAuth, (req: Request, res: Response) => {
  // Don't send passwordHash to client
  const user = req.user as any;
  const { passwordHash, ...userPublic } = user;
  res.json({ user: userPublic });
});

export const authRouter = router;
