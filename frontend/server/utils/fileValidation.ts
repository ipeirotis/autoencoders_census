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
 * - Text-decodable content (tolerates non-UTF-8 encodings like Windows-1252)
 * - Contains comma separators
 * - Has at least 2 rows (header + data)
 *
 * Encoding note: the worker-side `validate_csv()` uses chardet to detect
 * the actual encoding and then decodes accordingly, so non-UTF-8 CSVs
 * (e.g. Excel exports with smart quotes in Windows-1252) are fully
 * supported downstream. This Express-layer check intentionally does NOT
 * enforce UTF-8 — it only rejects files that are clearly binary (via
 * file-type magic-number detection) and verifies basic CSV structure so
 * that obviously-wrong uploads fail fast without burning a Pub/Sub +
 * worker round trip.
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

  // Decode as UTF-8 for the structural checks below. Non-UTF-8 bytes are
  // decoded as U+FFFD replacement characters, but that does NOT mean the
  // file is invalid — it just uses a different encoding (Windows-1252,
  // Latin-1, etc.) which the worker handles via chardet. We only reject
  // files that fail the structural checks (no commas, too few rows).
  let content: string;
  try {
    content = buffer.toString('utf-8');
  } catch {
    return { valid: false, reason: 'Unable to read file content' };
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
