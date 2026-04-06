/**
 * Authentication routes
 * Handles user signup, login, logout, and current user retrieval
 */

import { Router, Request, Response, NextFunction } from 'express';
import { passport, requireAuth } from '../middleware/auth';
import { validateSignup, validateLogin } from '../middleware/validation';
import {
  createUser,
  createVerificationToken,
  getUserByVerificationToken,
  setEmailVerified,
  getUserByEmail,
  createPasswordResetToken,
  getUserByResetToken,
  updatePassword,
} from '../models/user';
import { logger } from '../config/logger';
import { env } from '../config/env';

const router = Router();

/**
 * POST /api/auth/signup
 * Create a new user account
 */
router.post('/signup', validateSignup, async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;

    // Create user (validation already handled by middleware)
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
router.post('/login', validateLogin, (req: Request, res: Response, next: NextFunction) => {
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

/**
 * POST /api/auth/send-verification
 * Send email verification link
 * Email sending is stubbed (logs to console) in v1
 */
router.post('/send-verification', requireAuth, async (req: Request, res: Response) => {
  try {
    const user = req.user as any;

    if (user.emailVerified) {
      return res.status(400).json({ error: 'Email already verified' });
    }

    // Generate or regenerate verification token
    const token = user.verificationToken || await createVerificationToken(user.id);

    // STUB: In production, send actual email. For v1, log only.
    const verificationUrl = `${env.FRONTEND_URL}/verify-email?token=${token}`;
    logger.info('[EMAIL STUB] Verification link', { url: verificationUrl, userId: user.id });

    res.json({ message: 'Verification email sent' });
  } catch (error) {
    logger.error('Send verification failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    res.status(500).json({ error: 'Failed to send verification email' });
  }
});

/**
 * GET /api/auth/verify-email?token=xxx
 * Verify email with token
 */
router.get('/verify-email', async (req: Request, res: Response) => {
  try {
    const { token } = req.query;

    if (!token || typeof token !== 'string') {
      return res.status(400).json({ error: 'Token required' });
    }

    const user = await getUserByVerificationToken(token);
    if (!user) {
      return res.status(400).json({ error: 'Invalid or expired token' });
    }

    await setEmailVerified(user.id);

    logger.info('Email verified', { userId: user.id, email: user.email });
    res.json({ message: 'Email verified successfully' });
  } catch (error) {
    logger.error('Email verification failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    res.status(500).json({ error: 'Failed to verify email' });
  }
});

/**
 * POST /api/auth/request-reset
 * Request password reset link
 * Email sending is stubbed (logs to console) in v1
 * Always returns success to prevent email enumeration
 */
router.post('/request-reset', async (req: Request, res: Response) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    const user = await getUserByEmail(email);
    if (user) {
      const token = await createPasswordResetToken(user.id);

      // STUB: In production, send actual email. For v1, log only.
      const resetUrl = `${env.FRONTEND_URL}/reset-password?token=${token}`;
      logger.info('[EMAIL STUB] Password reset link', { url: resetUrl, userId: user.id });
    }

    // Always return success to prevent email enumeration
    res.json({
      message: 'If an account exists with that email, a reset link has been sent',
    });
  } catch (error) {
    logger.error('Request reset failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    res.status(500).json({ error: 'Failed to process password reset request' });
  }
});

/**
 * POST /api/auth/reset-password
 * Reset password with token
 */
router.post('/reset-password', async (req: Request, res: Response) => {
  try {
    const { token, newPassword } = req.body;

    if (!token || !newPassword) {
      return res.status(400).json({ error: 'Token and new password required' });
    }

    if (newPassword.length < 8) {
      return res.status(400).json({ error: 'Password must be at least 8 characters' });
    }

    const user = await getUserByResetToken(token);
    if (!user) {
      return res.status(400).json({ error: 'Invalid or expired reset token' });
    }

    await updatePassword(user.id, newPassword);

    logger.info('Password reset successful', { userId: user.id, email: user.email });
    res.json({ message: 'Password reset successfully' });
  } catch (error) {
    logger.error('Password reset failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    res.status(500).json({ error: 'Failed to reset password' });
  }
});

export const authRouter = router;
