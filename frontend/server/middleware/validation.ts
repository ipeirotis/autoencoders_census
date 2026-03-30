/**
 * Express-validator middleware chains for input validation
 * Validates all user inputs before they reach route handlers
 */

import { body, param, validationResult, ValidationChain } from 'express-validator';
import { Request, Response, NextFunction } from 'express';

/**
 * Middleware to handle validation errors
 * Returns 400 with field-level error details if validation fails
 * Calls next() if validation passes
 */
export function handleValidationErrors(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    res.status(400).json({
      error: 'Validation failed',
      details: errors.array().map((e) => ({
        field: e.type === 'field' ? e.path : e.type,
        message: e.msg,
      })),
    });
    return;
  }
  next();
}

/**
 * Validation chain for user signup
 * - Email must be valid format and normalized
 * - Password must be 8-128 characters
 */
export const validateSignup: (ValidationChain | typeof handleValidationErrors)[] = [
  body('email')
    .isEmail()
    .withMessage('Invalid email')
    .normalizeEmail()
    .trim(),
  body('password')
    .isLength({ min: 8, max: 128 })
    .withMessage('Password must be 8-128 characters'),
  handleValidationErrors,
];

/**
 * Validation chain for user login
 * - Email must be valid format and normalized
 * - Password must not be empty
 */
export const validateLogin: (ValidationChain | typeof handleValidationErrors)[] = [
  body('email')
    .isEmail()
    .withMessage('Invalid email')
    .normalizeEmail(),
  body('password')
    .notEmpty()
    .withMessage('Password required'),
  handleValidationErrors,
];

/**
 * Validation chain for job ID parameter
 * - Must be valid UUID format
 */
export const validateJobId: (ValidationChain | typeof handleValidationErrors)[] = [
  param('id')
    .isUUID()
    .withMessage('Invalid job ID format'),
  handleValidationErrors,
];

/**
 * Validation chain for upload URL request
 * - Filename required and must end with .csv
 * - Content type optional
 */
export const validateUploadUrl: (ValidationChain | typeof handleValidationErrors)[] = [
  body('filename')
    .notEmpty()
    .withMessage('Filename required'),
  body('filename')
    .matches(/\.csv$/i)
    .withMessage('Only CSV files allowed'),
  body('contentType')
    .optional()
    .isString(),
  handleValidationErrors,
];

/**
 * Validation chain for start job request
 * - Job ID must be valid UUID
 * - GCS filename required
 */
export const validateStartJob: (ValidationChain | typeof handleValidationErrors)[] = [
  body('jobId')
    .isUUID()
    .withMessage('Invalid job ID'),
  body('gcsFileName')
    .notEmpty()
    .withMessage('GCS filename required'),
  handleValidationErrors,
];
