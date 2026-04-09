/**
 * File validation utilities
 * CSV content validation and safe filename generation
 */

import { fileTypeFromBuffer } from 'file-type';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';

/**
 * Validates that a buffer contains valid CSV content
 * Checks for:
 * - Not a known binary format (via file-type detection)
 * - Valid UTF-8 encoding
 * - Contains comma separators
 * - Has at least 2 rows (header + data)
 *
 * @param buffer - File content as Buffer
 * @returns {valid: true} or {valid: false, reason: string}
 */
export async function validateCSVContent(
  buffer: Buffer
): Promise<{ valid: boolean; reason?: string }> {
  // Check if it's a known binary format (not CSV)
  const type = await fileTypeFromBuffer(buffer);
  if (type) {
    return { valid: false, reason: 'File appears to be binary, not CSV' };
  }

  // Try to decode as UTF-8
  let content: string;
  try {
    content = buffer.toString('utf-8');
    // Check for invalid UTF-8 sequences (replacement character)
    if (content.includes('\ufffd')) {
      return { valid: false, reason: 'Invalid file encoding' };
    }
  } catch {
    return { valid: false, reason: 'Invalid file encoding' };
  }

  // Check CSV structure
  const lines = content.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) {
    return { valid: false, reason: 'CSV must have header and at least one data row' };
  }

  if (!content.includes(',')) {
    return { valid: false, reason: 'File does not appear to be comma-separated' };
  }

  return { valid: true };
}

/**
 * Generates a safe filename using UUID
 * User-provided filenames are never used - only UUID is used
 * Format: uploads/{userId}/{uuid}.csv
 *
 * @param userId - User ID for path organization
 * @returns Safe filename path
 */
export function generateSafeFilename(userId: string): string {
  const fileId = uuidv4();
  return `uploads/${userId}/${fileId}.csv`;
}

/**
 * Sanitizes a file path to prevent path traversal attacks
 * Ensures the resolved path stays within the upload directory
 *
 * @param uploadDir - Base upload directory (absolute path)
 * @param unsafePath - User-provided path (may contain traversal attempts)
 * @returns Sanitized absolute path if safe, null if traversal detected
 */
export function sanitizePath(uploadDir: string, unsafePath: string): string | null {
  const normalizedUploadDir = path.resolve(uploadDir);
  const resolved = path.resolve(normalizedUploadDir, unsafePath);

  // Verify resolved path is within the upload directory.
  // Use a trailing separator so sibling directories that share the same
  // prefix (e.g. `/var/uploads_evil/...` vs `/var/uploads`) are rejected.
  // Also accept an exact match for the directory itself.
  if (
    resolved !== normalizedUploadDir &&
    !resolved.startsWith(normalizedUploadDir + path.sep)
  ) {
    return null; // Path traversal attempt
  }

  return resolved;
}
