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
 * Validation chain for password reset request
 * - Email must be a valid format and normalized
 * Mirrors signup/login validation so malformed payloads (e.g. non-string
 * email) return a consistent 400 instead of crashing downstream string ops.
 */
export const validateRequestReset: (ValidationChain | typeof handleValidationErrors)[] = [
  body('email')
    .isEmail()
    .withMessage('Invalid email')
    .normalizeEmail(),
  handleValidationErrors,
];

/**
 * Validation chain for password reset confirmation
 * - Token must be a non-empty string
 * - newPassword must be a string of 8-128 characters
 * Strict string checks so non-string payloads (e.g. `{ newPassword: {} }`)
 * are rejected with 400 instead of bypassing the length guard and crashing
 * inside bcrypt with a 500.
 */
export const validateResetPassword: (ValidationChain | typeof handleValidationErrors)[] = [
  body('token')
    .isString()
    .withMessage('Token required')
    .bail()
    .notEmpty()
    .withMessage('Token required'),
  body('newPassword')
    .isString()
    .withMessage('Password must be a string')
    .bail()
    .isLength({ min: 8, max: 128 })
    .withMessage('Password must be 8-128 characters'),
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
 * Allowed model preset ids (TASKS.md 3.2). Must stay in sync with
 * `VALID_PRESETS` in `model/presets.py`. Anything outside this list is
 * coerced to `auto` by the worker, but we reject it at the API layer
 * too so an invalid value never reaches Pub/Sub.
 *
 * NOTE: model/presets.py and frontend/server/utils/modelPresets.ts both
 * mirror this list. When adding a preset, update all three call sites.
 */
export const VALID_MODEL_PRESETS = ['auto', 'small', 'medium', 'large'] as const;

/**
 * Validation chain for start job request
 * - Job ID must be valid UUID
 * - GCS filename required
 * - modelPreset is optional and must be one of VALID_MODEL_PRESETS
 *   when present (TASKS.md 3.2)
 */
export const validateStartJob: (ValidationChain | typeof handleValidationErrors)[] = [
  body('jobId')
    .isUUID()
    .withMessage('Invalid job ID'),
  body('gcsFileName')
    .notEmpty()
    .withMessage('GCS filename required'),
  body('modelPreset')
    .optional({ nullable: true })
    .isString()
    .withMessage('Model preset must be a string')
    .bail()
    .isIn(VALID_MODEL_PRESETS as unknown as string[])
    .withMessage(
      `Model preset must be one of: ${VALID_MODEL_PRESETS.join(', ')}`
    ),
  handleValidationErrors,
];
