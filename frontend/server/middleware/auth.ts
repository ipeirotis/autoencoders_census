/**
 * Passport authentication configuration and middleware
 * Implements local strategy for email/password authentication
 */

import passport from 'passport';
import { Strategy as LocalStrategy } from 'passport-local';
import bcrypt from 'bcrypt';
import type { Request, Response, NextFunction } from 'express';
import { getUserByEmail, getUserById } from '../models/user';

/**
 * Configure Passport local strategy
 * Validates email and password against Firestore user records
 */
passport.use(
  new LocalStrategy(
    {
      usernameField: 'email',
      passwordField: 'password',
    },
    async (email, password, done) => {
      try {
        // Look up user by email
        const user = await getUserByEmail(email);
        if (!user) {
          return done(null, false, { message: 'Invalid credentials' });
        }

        // Compare password with stored hash
        const isValid = await bcrypt.compare(password, user.passwordHash);
        if (!isValid) {
          return done(null, false, { message: 'Invalid credentials' });
        }

        // Authentication successful
        return done(null, user);
      } catch (error) {
        return done(error);
      }
    }
  )
);

/**
 * Serialize user to session
 * Stores only user ID in session for efficiency
 */
passport.serializeUser((user: any, done) => {
  done(null, user.id);
});

/**
 * Deserialize user from session
 * Fetches full user object from Firestore using stored ID
 */
passport.deserializeUser(async (id: string, done) => {
  try {
    const user = await getUserById(id);
    if (!user) {
      return done(null, false);
    }
    done(null, user);
  } catch (error) {
    done(error);
  }
});

/**
 * Middleware to require authentication
 * Returns 401 if user is not authenticated
 */
export const requireAuth = (req: Request, res: Response, next: NextFunction) => {
  if (req.isAuthenticated()) {
    return next();
  }
  res.status(401).json({ error: 'Authentication required' });
};

/**
 * Export configured passport instance
 */
export { passport };
