/**
 * Session configuration with Firestore store
 * Provides secure session management with Firestore backend
 */

import session from 'express-session';
import { Firestore } from '@google-cloud/firestore';
import firestoreStoreFactory from 'firestore-store';
import { env } from './env';

// Initialize Firestore for sessions
const firestore = new Firestore();

// Create FirestoreStore class
const FirestoreStore = firestoreStoreFactory(session);

/**
 * Session middleware configuration
 * Uses Firestore for session persistence across server restarts
 * Cookies are secure in production, 24-hour expiry
 *
 * SameSite policy:
 * - Production uses `sameSite=none` + `secure=true` so the session cookie is
 *   attached on cross-site `fetch`/XHR requests when the frontend (e.g.
 *   *.vercel.app) and API (e.g. Cloud Run) are served from different sites.
 *   CSRF protection is provided by the CORS origin whitelist in
 *   `middleware/security.ts`, which only allows credentialed requests from
 *   FRONTEND_URL.
 * - Development uses `sameSite=lax` because `none` requires `secure=true`,
 *   which browsers refuse over plain HTTP (localhost dev).
 */
const isProduction = env.NODE_ENV === 'production';

export const sessionConfig = session({
  store: new FirestoreStore({
    database: firestore,
    collection: 'sessions',
  }),
  secret: env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: isProduction, // HTTPS only in production (required for sameSite=none)
    httpOnly: true, // Prevent client-side JavaScript access
    sameSite: isProduction ? 'none' : 'lax', // Allow cross-site auth in prod
    maxAge: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
  },
});
