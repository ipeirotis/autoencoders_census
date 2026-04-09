/**
 * User model with Firestore operations
 * Handles user authentication, email verification, and password reset
 */

import { Firestore } from '@google-cloud/firestore';
import bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';
import crypto from 'crypto';

// Initialize Firestore
const firestore = new Firestore();
const USERS_COLLECTION = 'users';
/**
 * Firestore collection used as a unique-email index. Document ID is the
 * normalized email, value is `{ userId }`. Writing to this collection inside
 * the same transaction that creates the user document gives us atomic email
 * uniqueness: two concurrent signups for the same email cannot both commit.
 */
const EMAIL_INDEX_COLLECTION = 'email_index';
const BCRYPT_ROUNDS = 12;

/**
 * Normalize an email for storage and for use as the `email_index` document
 * key. Kept simple (lowercase + trim) so it is stable and deterministic;
 * Firestore document IDs allow `@` and `.`, so no escaping is needed.
 */
function normalizeEmail(email: string): string {
  return email.toLowerCase().trim();
}

/**
 * User interface (without passwordHash, but still contains sensitive fields
 * like verificationToken/resetToken that must never be sent to API clients)
 */
export interface User {
  id: string;
  email: string;
  createdAt: string;
  emailVerified: boolean;
  verificationToken?: string;
  resetToken?: string;
  resetTokenExpiry?: string;
}

/**
 * Internal user interface (includes passwordHash)
 */
interface UserInternal extends User {
  passwordHash: string;
}

/**
 * Public user interface - safe to return in API responses.
 * Excludes passwordHash, verificationToken, resetToken, and resetTokenExpiry
 * so clients can never read credentials or verification/reset secrets.
 */
export interface PublicUser {
  id: string;
  email: string;
  createdAt: string;
  emailVerified: boolean;
}

/**
 * Strip all sensitive fields from a user object before returning it to clients.
 * Removes passwordHash, verificationToken, resetToken, and resetTokenExpiry.
 */
export function toPublicUser(user: User | UserInternal): PublicUser {
  return {
    id: user.id,
    email: user.email,
    createdAt: user.createdAt,
    emailVerified: user.emailVerified,
  };
}

/**
 * Create a new user with hashed password.
 *
 * Email uniqueness is enforced atomically using a Firestore transaction and
 * a dedicated `email_index` collection keyed by the normalized email. The
 * transaction reads the email index doc and, only if it does not exist,
 * writes both the user document and the email index doc in a single commit.
 * Two concurrent signups for the same email therefore cannot both succeed:
 * the second transaction will see the first write and throw.
 *
 * @param email - User's email address
 * @param password - Plain text password (will be hashed)
 * @returns User object without passwordHash
 * @throws Error if email already exists
 */
export async function createUser(email: string, password: string): Promise<User> {
  const normalizedEmail = normalizeEmail(email);

  // Generate user ID and hash password up front (hashing is CPU-bound, keep
  // it outside the transaction so we don't hold the transaction open).
  const userId = uuidv4();
  const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
  const verificationToken = crypto.randomBytes(32).toString('hex');

  const userData: UserInternal = {
    id: userId,
    email: normalizedEmail,
    passwordHash,
    createdAt: new Date().toISOString(),
    emailVerified: false,
    verificationToken,
  };

  const userRef = firestore.collection(USERS_COLLECTION).doc(userId);
  const emailRef = firestore.collection(EMAIL_INDEX_COLLECTION).doc(normalizedEmail);

  await firestore.runTransaction(async (tx) => {
    const existing = await tx.get(emailRef);
    if (existing.exists) {
      throw new Error('Email already registered');
    }
    tx.set(userRef, userData);
    tx.set(emailRef, { userId, createdAt: userData.createdAt });
  });

  // Return user without passwordHash
  const { passwordHash: _, ...userPublic } = userData;
  return userPublic;
}

/**
 * Get user by email address
 * @param email - User's email address
 * @returns User object (with passwordHash) or null if not found
 */
export async function getUserByEmail(email: string): Promise<UserInternal | null> {
  const snapshot = await firestore
    .collection(USERS_COLLECTION)
    .where('email', '==', normalizeEmail(email))
    .limit(1)
    .get();

  if (snapshot.empty) {
    return null;
  }

  const doc = snapshot.docs[0];
  return { id: doc.id, ...doc.data() } as UserInternal;
}

/**
 * Get user by ID
 * @param userId - User's ID
 * @returns User object (with passwordHash) or null if not found
 */
export async function getUserById(userId: string): Promise<UserInternal | null> {
  const doc = await firestore.collection(USERS_COLLECTION).doc(userId).get();

  if (!doc.exists) {
    return null;
  }

  return { id: doc.id, ...doc.data() } as UserInternal;
}

/**
 * Mark user's email as verified
 * @param userId - User's ID
 */
export async function setEmailVerified(userId: string): Promise<void> {
  await firestore.collection(USERS_COLLECTION).doc(userId).update({
    emailVerified: true,
    verificationToken: null,
  });
}

/**
 * Create password reset token with 1-hour expiry
 * @param userId - User's ID
 * @returns Reset token
 */
export async function createPasswordResetToken(userId: string): Promise<string> {
  const resetToken = crypto.randomBytes(32).toString('hex');
  const resetTokenExpiry = new Date(Date.now() + 3600000).toISOString(); // 1 hour from now

  await firestore.collection(USERS_COLLECTION).doc(userId).update({
    resetToken,
    resetTokenExpiry,
  });

  return resetToken;
}

/**
 * Get user by reset token (only if not expired)
 * @param token - Password reset token
 * @returns User object or null if token invalid/expired
 */
export async function getUserByResetToken(token: string): Promise<UserInternal | null> {
  const snapshot = await firestore
    .collection(USERS_COLLECTION)
    .where('resetToken', '==', token)
    .limit(1)
    .get();

  if (snapshot.empty) {
    return null;
  }

  const doc = snapshot.docs[0];
  const user = { id: doc.id, ...doc.data() } as UserInternal;

  // Check if token is expired
  if (user.resetTokenExpiry && new Date(user.resetTokenExpiry) < new Date()) {
    return null;
  }

  return user;
}

/**
 * Update user's password and clear reset token
 * @param userId - User's ID
 * @param newPassword - New plain text password (will be hashed)
 */
export async function updatePassword(userId: string, newPassword: string): Promise<void> {
  const passwordHash = await bcrypt.hash(newPassword, BCRYPT_ROUNDS);

  await firestore.collection(USERS_COLLECTION).doc(userId).update({
    passwordHash,
    resetToken: null,
    resetTokenExpiry: null,
  });
}

/**
 * Create or regenerate verification token for user
 * @param userId - User's ID
 * @returns Verification token
 */
export async function createVerificationToken(userId: string): Promise<string> {
  const verificationToken = crypto.randomBytes(32).toString('hex');

  await firestore.collection(USERS_COLLECTION).doc(userId).update({
    verificationToken,
  });

  return verificationToken;
}

/**
 * Get user by verification token
 * @param token - Email verification token
 * @returns User object or null if token invalid
 */
export async function getUserByVerificationToken(token: string): Promise<UserInternal | null> {
  const snapshot = await firestore
    .collection(USERS_COLLECTION)
    .where('verificationToken', '==', token)
    .limit(1)
    .get();

  if (snapshot.empty) {
    return null;
  }

  const doc = snapshot.docs[0];
  return { id: doc.id, ...doc.data() } as UserInternal;
}
