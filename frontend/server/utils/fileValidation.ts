/**
 * File validation utilities
 * CSV content validation and safe filename generation
 */

import { fileTypeFromBuffer } from 'file-type';

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
