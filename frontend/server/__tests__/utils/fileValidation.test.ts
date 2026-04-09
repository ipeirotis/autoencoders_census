/**
 * Tests for file validation utilities
 * CSV content validation and safe filename generation
 */

import { describe, it, expect } from '@jest/globals';
import path from 'path';
import {
  validateCSVContent,
  generateSafeFilename,
  sanitizePath,
} from '../../utils/fileValidation';

describe('CSV content validation', () => {
  describe('validateCSVContent', () => {
    it('should accept valid CSV (header + data rows with commas)', async () => {
      const validCSV = Buffer.from('name,age,city\nJohn,30,NYC\nJane,25,LA\n');
      const result = await validateCSVContent(validCSV);

      expect(result.valid).toBe(true);
      expect(result.reason).toBeUndefined();
    });

    it('should reject binary file renamed to .csv', async () => {
      // PNG file signature
      const pngBuffer = Buffer.from([
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
      ]);
      const result = await validateCSVContent(pngBuffer);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('binary');
    });

    it('should reject file with no commas', async () => {
      const noCommas = Buffer.from('name age city\nJohn 30 NYC\nJane 25 LA\n');
      const result = await validateCSVContent(noCommas);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('comma');
    });

    it('should reject file with only 1 row (no data)', async () => {
      const headerOnly = Buffer.from('name,age,city\n');
      const result = await validateCSVContent(headerOnly);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('header and at least one data row');
    });

    it('should reject non-UTF8 encoding', async () => {
      // Invalid UTF-8 sequence (also detected as binary by file-type)
      const invalidUtf8 = Buffer.from([0xff, 0xfe, 0xfd]);
      const result = await validateCSVContent(invalidUtf8);

      expect(result.valid).toBe(false);
      // file-type detects this as binary before UTF-8 check
      expect(result.reason).toMatch(/binary|encoding/);
    });

    it('should return { valid: true } for valid CSV', async () => {
      const validCSV = Buffer.from('id,value\n1,test\n2,data\n');
      const result = await validateCSVContent(validCSV);

      expect(result).toEqual({ valid: true });
    });

    it('should return { valid: false, reason: string } for invalid CSV', async () => {
      const invalidCSV = Buffer.from('not,csv\n');
      const result = await validateCSVContent(invalidCSV);

      expect(result).toHaveProperty('valid', false);
      expect(result).toHaveProperty('reason');
      expect(typeof result.reason).toBe('string');
    });

    it('should accept CSV with Windows line endings (CRLF)', async () => {
      const csvWithCRLF = Buffer.from('name,age\r\nJohn,30\r\nJane,25\r\n');
      const result = await validateCSVContent(csvWithCRLF);

      expect(result.valid).toBe(true);
    });

    it('should reject ZIP file renamed to .csv', async () => {
      // ZIP file signature
      const zipBuffer = Buffer.from([0x50, 0x4b, 0x03, 0x04]);
      const result = await validateCSVContent(zipBuffer);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('binary');
    });

    it('should reject PDF file renamed to .csv', async () => {
      // PDF file signature
      const pdfBuffer = Buffer.from('%PDF-1.4\n');
      const result = await validateCSVContent(pdfBuffer);

      expect(result.valid).toBe(false);
      expect(result.reason).toContain('binary');
    });

    it('should accept CSV with empty cells', async () => {
      const csvWithEmpty = Buffer.from('name,age,city\nJohn,,NYC\n,25,LA\n');
      const result = await validateCSVContent(csvWithEmpty);

      expect(result.valid).toBe(true);
    });

    it('should accept CSV with quoted fields', async () => {
      const csvWithQuotes = Buffer.from(
        'name,description\n"John","A person"\n"Jane","Another person"\n'
      );
      const result = await validateCSVContent(csvWithQuotes);

      expect(result.valid).toBe(true);
    });
  });
});

describe('Path traversal protection', () => {
  describe('generateSafeFilename', () => {
    it('should return UUID-based path', () => {
      const userId = 'user123';
      const filename = generateSafeFilename(userId);

      expect(filename).toMatch(/^uploads\/user123\/[a-f0-9-]{36}\.csv$/);
    });

    it('should ignore user-provided filename (not passed to function)', () => {
      // generateSafeFilename only takes userId, generates its own UUID
      const userId = 'user456';
      const filename1 = generateSafeFilename(userId);
      const filename2 = generateSafeFilename(userId);

      // Different UUIDs each time
      expect(filename1).not.toBe(filename2);
      expect(filename1).toMatch(/^uploads\/user456\//);
      expect(filename2).toMatch(/^uploads\/user456\//);
    });

    it('should always use .csv extension', () => {
      const userId = 'user789';
      const filename = generateSafeFilename(userId);

      expect(filename).toMatch(/\.csv$/);
    });

    it('should include userId in path', () => {
      const userId = 'test-user';
      const filename = generateSafeFilename(userId);

      expect(filename).toContain('uploads/test-user/');
    });
  });

  describe('sanitizePath', () => {
    it('should reject "../" in path', () => {
      const uploadDir = '/var/uploads';
      const maliciousPath = '../etc/passwd';

      const result = sanitizePath(uploadDir, maliciousPath);

      expect(result).toBeNull();
    });

    it('should reject absolute paths outside upload dir', () => {
      const uploadDir = '/var/uploads';
      const absolutePath = '/etc/passwd';

      const result = sanitizePath(uploadDir, absolutePath);

      expect(result).toBeNull();
    });

    it('should normalize path with path.resolve', () => {
      const uploadDir = '/var/uploads';
      const safePath = 'user123/file.csv';

      const result = sanitizePath(uploadDir, safePath);

      expect(result).toBe(path.resolve(uploadDir, safePath));
      expect(result).toBe('/var/uploads/user123/file.csv');
    });

    it('should return null for traversal attempts', () => {
      const uploadDir = '/var/uploads';
      const traversalPaths = [
        '../../etc/passwd',
        '../../../root/.ssh/id_rsa',
        'user/../../../etc/passwd',
      ];

      traversalPaths.forEach((maliciousPath) => {
        const result = sanitizePath(uploadDir, maliciousPath);
        expect(result).toBeNull();
      });
    });

    it('should accept safe relative paths', () => {
      const uploadDir = '/var/uploads';
      const safePath = 'user123/data/file.csv';

      const result = sanitizePath(uploadDir, safePath);

      expect(result).not.toBeNull();
      expect(result).toContain('/var/uploads/');
    });

    it('should handle Windows-style paths (on Unix)', () => {
      const uploadDir = '/var/uploads';
      const windowsPath = 'user123\\..\\..\\etc\\passwd';

      const result = sanitizePath(uploadDir, windowsPath);

      // On Unix, backslashes are valid filename characters
      // But the path should still be within uploadDir
      expect(result).not.toBeNull();
      if (result) {
        expect(result.startsWith(path.resolve(uploadDir))).toBe(true);
      }
    });
  });
});
