/**
 * Sanitizes cell values to prevent Excel formula injection attacks.
 *
 * Per OWASP recommendation, any cell starting with dangerous characters
 * is prefixed with a single quote to force Excel to treat it as text.
 *
 * @param value - The cell value to sanitize
 * @returns Sanitized value (string prefixed with ' if dangerous, otherwise unchanged)
 */
export function sanitizeFormulaInjection(value: any): any {
  // Handle null/undefined
  if (value == null) {
    return value;
  }

  // Only sanitize strings
  if (typeof value !== 'string') {
    return value;
  }

  // Dangerous characters that can trigger formula execution in Excel
  const dangerousChars = ['=', '+', '-', '@', '\t', '\r'];

  // Check if string starts with any dangerous character
  if (dangerousChars.some(char => value.startsWith(char))) {
    return `'${value}`;
  }

  return value;
}
