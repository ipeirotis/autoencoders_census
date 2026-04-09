/**
 * Tests for User model with Firestore operations
 */

import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import type { Firestore, DocumentReference, DocumentSnapshot, Query, QuerySnapshot } from '@google-cloud/firestore';

// Mock Firestore before importing user model
const mockSet = jest.fn().mockResolvedValue(undefined);
const mockGet = jest.fn();
const mockUpdate = jest.fn().mockResolvedValue(undefined);
const mockWhere = jest.fn();
const mockLimit = jest.fn();
const mockDoc = jest.fn();
const mockCollection = jest.fn();

// Transaction-scoped mocks used by createUser's runTransaction block
const mockTxGet = jest.fn();
const mockTxSet = jest.fn();
const mockRunTransaction = jest.fn();

// Mock Firestore constructor
jest.mock('@google-cloud/firestore', () => {
  return {
    Firestore: jest.fn().mockImplementation(() => ({
      collection: mockCollection,
      runTransaction: mockRunTransaction,
    })),
  };
});

describe('User model', () => {
  let createUser: any;
  let getUserByEmail: any;
  let getUserById: any;
  let setEmailVerified: any;
  let createPasswordResetToken: any;
  let getUserByResetToken: any;
  let updatePassword: any;

  beforeEach(async () => {
    jest.clearAllMocks();
    jest.resetModules();

    // Setup mock chain for collection().doc()
    mockDoc.mockReturnValue({
      set: mockSet,
      get: mockGet,
      update: mockUpdate,
    });

    mockCollection.mockReturnValue({
      doc: mockDoc,
      where: mockWhere,
    });

    // Setup mock chain for where().limit()
    mockLimit.mockReturnValue({
      get: mockGet,
    });

    mockWhere.mockReturnValue({
      limit: mockLimit,
      get: mockGet,
    });

    // runTransaction() invokes the callback with a transaction object that
    // exposes tx.get / tx.set. We forward only the data argument to the
    // top-level mockSet so existing assertions on `mockSet.mock.calls[i][0]`
    // (which treat the first argument as the saved data) keep working for
    // the new createUser path. Tests program mockTxGet to control the
    // "email already exists" branch.
    mockRunTransaction.mockImplementation(async (callback: any) => {
      return callback({
        get: mockTxGet,
        set: (_ref: any, data: any) => mockSet(data),
      });
    });

    // Import user model after mocks are set up
    const userModule = await import('../../models/user');
    createUser = userModule.createUser;
    getUserByEmail = userModule.getUserByEmail;
    getUserById = userModule.getUserById;
    setEmailVerified = userModule.setEmailVerified;
    createPasswordResetToken = userModule.createPasswordResetToken;
    getUserByResetToken = userModule.getUserByResetToken;
    updatePassword = userModule.updatePassword;
  });

  describe('createUser', () => {
    it('should store user with hashed password (not plaintext)', async () => {
      const email = 'test@example.com';
      const password = 'TestPassword123';

      // Mock the in-transaction read of the email_index doc: email doesn't exist
      mockTxGet.mockResolvedValueOnce({ exists: false });

      const user = await createUser(email, password);

      expect(mockSet).toHaveBeenCalled();
      const savedData = mockSet.mock.calls[0][0];
      expect(savedData.passwordHash).toBeDefined();
      expect(savedData.passwordHash).not.toBe(password);
      expect(savedData.passwordHash.length).toBeGreaterThan(20); // bcrypt hashes are long
    });

    it('should return user object without passwordHash', async () => {
      // Mock the in-transaction read of the email_index doc: email doesn't exist
      mockTxGet.mockResolvedValueOnce({ exists: false });

      const user = await createUser('test@example.com', 'password123');

      expect(user.email).toBe('test@example.com');
      expect(user.id).toBeDefined();
      expect(user.createdAt).toBeDefined();
      expect(user.emailVerified).toBe(false);
      expect(user.verificationToken).toBeDefined();
      expect((user as any).passwordHash).toBeUndefined(); // Should not be exposed
    });

    it('should throw if email already exists', async () => {
      // Mock the in-transaction read of the email_index doc: email exists
      mockTxGet.mockResolvedValueOnce({ exists: true });

      await expect(createUser('test@example.com', 'password123'))
        .rejects
        .toThrow('Email already registered');
    });

    it('should write both user doc and email_index doc atomically', async () => {
      mockTxGet.mockResolvedValueOnce({ exists: false });

      await createUser('test@example.com', 'password123');

      // Expect two tx.set() calls: one for the user doc, one for the email index
      expect(mockSet).toHaveBeenCalledTimes(2);
      const userDocData = mockSet.mock.calls[0][0] as any;
      const emailIndexData = mockSet.mock.calls[1][0] as any;
      expect(userDocData.email).toBe('test@example.com');
      expect(userDocData.passwordHash).toBeDefined();
      expect(emailIndexData.userId).toBe(userDocData.id);
    });
  });

  describe('getUserByEmail', () => {
    it('should return user when email exists', async () => {
      const mockUserData = {
        email: 'test@example.com',
        passwordHash: 'hashedpassword',
        createdAt: new Date().toISOString(),
        emailVerified: false,
      };

      mockGet.mockResolvedValueOnce({
        empty: false,
        docs: [{
          id: 'user-123',
          data: () => mockUserData,
        }],
      });

      const user = await getUserByEmail('test@example.com');

      expect(user).toBeDefined();
      expect(user?.id).toBe('user-123');
      expect(user?.email).toBe('test@example.com');
      expect(user?.passwordHash).toBe('hashedpassword');
    });

    it('should return null when email not found', async () => {
      mockGet.mockResolvedValueOnce({
        empty: true,
        docs: [],
      });

      const user = await getUserByEmail('nonexistent@example.com');

      expect(user).toBeNull();
    });
  });

  describe('getUserById', () => {
    it('should return user when ID exists', async () => {
      const mockUserData = {
        email: 'test@example.com',
        passwordHash: 'hashedpassword',
        createdAt: new Date().toISOString(),
        emailVerified: false,
      };

      mockGet.mockResolvedValueOnce({
        exists: true,
        id: 'user-123',
        data: () => mockUserData,
      });

      const user = await getUserById('user-123');

      expect(user).toBeDefined();
      expect(user?.id).toBe('user-123');
      expect(user?.email).toBe('test@example.com');
    });

    it('should return null when ID not found', async () => {
      mockGet.mockResolvedValueOnce({
        exists: false,
      });

      const user = await getUserById('nonexistent-id');

      expect(user).toBeNull();
    });
  });

  describe('setEmailVerified', () => {
    it('should update user emailVerified field to true', async () => {
      await setEmailVerified('user-123');

      expect(mockUpdate).toHaveBeenCalledWith({
        emailVerified: true,
        verificationToken: null,
      });
    });
  });

  describe('createPasswordResetToken', () => {
    it('should generate token and store with expiry', async () => {
      const token = await createPasswordResetToken('user-123');

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.length).toBeGreaterThan(10);

      expect(mockUpdate).toHaveBeenCalled();
      const updateData = mockUpdate.mock.calls[0][0];
      expect(updateData.resetToken).toBe(token);
      expect(updateData.resetTokenExpiry).toBeDefined();
      expect(new Date(updateData.resetTokenExpiry).getTime()).toBeGreaterThan(Date.now());
    });
  });

  describe('getUserByResetToken', () => {
    it('should return user if token valid and not expired', async () => {
      const futureExpiry = new Date(Date.now() + 3600000).toISOString(); // 1 hour from now
      const mockUserData = {
        email: 'test@example.com',
        passwordHash: 'hashedpassword',
        resetToken: 'valid-token',
        resetTokenExpiry: futureExpiry,
      };

      mockGet.mockResolvedValueOnce({
        empty: false,
        docs: [{
          id: 'user-123',
          data: () => mockUserData,
        }],
      });

      const user = await getUserByResetToken('valid-token');

      expect(user).toBeDefined();
      expect(user?.id).toBe('user-123');
      expect(user?.resetToken).toBe('valid-token');
    });

    it('should return null if token expired', async () => {
      const pastExpiry = new Date(Date.now() - 3600000).toISOString(); // 1 hour ago
      const mockUserData = {
        email: 'test@example.com',
        resetToken: 'expired-token',
        resetTokenExpiry: pastExpiry,
      };

      mockGet.mockResolvedValueOnce({
        empty: false,
        docs: [{
          id: 'user-123',
          data: () => mockUserData,
        }],
      });

      const user = await getUserByResetToken('expired-token');

      expect(user).toBeNull();
    });
  });

  describe('updatePassword', () => {
    it('should update passwordHash for user', async () => {
      const newPassword = 'NewPassword123';

      await updatePassword('user-123', newPassword);

      expect(mockUpdate).toHaveBeenCalled();
      const updateData = mockUpdate.mock.calls[0][0];
      expect(updateData.passwordHash).toBeDefined();
      expect(updateData.passwordHash).not.toBe(newPassword); // Should be hashed
      expect(updateData.resetToken).toBeNull();
      expect(updateData.resetTokenExpiry).toBeNull();
    });
  });
});
