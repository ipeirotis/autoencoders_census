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
 */
export const sessionConfig = session({
  store: new FirestoreStore({
    database: firestore,
    collection: 'sessions',
  }),
  secret: env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: env.NODE_ENV === 'production', // HTTPS only in production
    httpOnly: true, // Prevent client-side JavaScript access
    sameSite: 'lax', // CSRF protection
    maxAge: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
  },
});
