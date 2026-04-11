/**
 * Helpers for rendering job errors (TASKS.md 2.3).
 *
 * The worker writes a structured error payload to Firestore:
 *   - error: human-readable message
 *   - errorCode: stable machine-readable code (e.g. "csv_too_large")
 *   - errorType: pipeline-stage bucket
 *       ("validation" | "processing" | "training" | "scoring" | "internal")
 *
 * These helpers map those codes into a heading + fallback user-facing
 * message so the UI can show a clean, consistent error state regardless of
 * whether the worker wrote a message or not. Keeping this logic in one
 * place makes it easy to keep parity between Index.tsx's inline error
 * banner and JobProgress.tsx's full error card.
 */

/** Stable error-code constants mirroring worker.ErrorCode. */
export const JOB_ERROR_CODES = {
  CSV_TOO_LARGE: 'csv_too_large',
  CSV_EMPTY: 'csv_empty',
  CSV_ENCODING: 'csv_encoding',
  CSV_PARSE: 'csv_parse',
  CSV_INCONSISTENT_COLUMNS: 'csv_inconsistent_columns',
  CSV_TOO_FEW_ROWS: 'csv_too_few_rows',
  CSV_TOO_FEW_COLUMNS: 'csv_too_few_columns',
  LOAD_FAILURE: 'load_failure',
  NO_USABLE_COLUMNS: 'no_usable_columns',
  TRAINING_FAILURE: 'training_failure',
  SCORING_FAILURE: 'scoring_failure',
  INTERNAL_ERROR: 'internal_error',
} as const;

export type JobErrorCode = (typeof JOB_ERROR_CODES)[keyof typeof JOB_ERROR_CODES];

export type JobErrorType =
  | 'validation'
  | 'processing'
  | 'training'
  | 'scoring'
  | 'internal';

/** Per-code heading + user-facing fallback message. */
const ERROR_COPY: Record<
  string,
  { heading: string; fallbackMessage: string }
> = {
  [JOB_ERROR_CODES.CSV_TOO_LARGE]: {
    heading: 'File too large',
    fallbackMessage:
      'The uploaded file exceeds the 100 MB limit. Please upload a smaller CSV.',
  },
  [JOB_ERROR_CODES.CSV_EMPTY]: {
    heading: 'CSV file is empty',
    fallbackMessage:
      'The uploaded file has no data. Please upload a CSV with at least a header and some rows.',
  },
  [JOB_ERROR_CODES.CSV_ENCODING]: {
    heading: 'Unsupported file encoding',
    fallbackMessage:
      'We could not read this file. Please save it as UTF-8 or a standard Windows/Latin-1 CSV and try again.',
  },
  [JOB_ERROR_CODES.CSV_PARSE]: {
    heading: 'Could not parse CSV',
    fallbackMessage:
      'The file is not a valid CSV. Please check it opens cleanly in a spreadsheet tool and re-upload.',
  },
  [JOB_ERROR_CODES.CSV_INCONSISTENT_COLUMNS]: {
    heading: 'Inconsistent row length',
    fallbackMessage:
      'Some rows have a different number of columns than the header. Please fix the file and re-upload.',
  },
  [JOB_ERROR_CODES.CSV_TOO_FEW_ROWS]: {
    heading: 'Not enough rows',
    fallbackMessage:
      'The CSV must contain at least 10 data rows for training to work.',
  },
  [JOB_ERROR_CODES.CSV_TOO_FEW_COLUMNS]: {
    heading: 'Not enough columns',
    fallbackMessage:
      'The CSV must contain at least 2 columns.',
  },
  [JOB_ERROR_CODES.LOAD_FAILURE]: {
    heading: 'Could not load data',
    fallbackMessage:
      'The file looked like a CSV but could not be loaded. Please check the format and try again.',
  },
  [JOB_ERROR_CODES.NO_USABLE_COLUMNS]: {
    heading: 'No usable columns',
    fallbackMessage:
      'Every column was dropped because it had only one value or more than 9 distinct values. Try a dataset with some low-cardinality categorical columns.',
  },
  [JOB_ERROR_CODES.TRAINING_FAILURE]: {
    heading: 'Model training failed',
    fallbackMessage:
      'We could not train a model on this dataset. It may be too small or too uniform to detect outliers.',
  },
  [JOB_ERROR_CODES.SCORING_FAILURE]: {
    heading: 'Scoring failed',
    fallbackMessage:
      'The model was trained but outlier scoring failed. Please try again.',
  },
  [JOB_ERROR_CODES.INTERNAL_ERROR]: {
    heading: 'Unexpected error',
    fallbackMessage:
      'Something went wrong on our side. Please try again or contact support if the problem persists.',
  },
};

const TYPE_HEADINGS: Record<JobErrorType, string> = {
  validation: 'Invalid file',
  processing: 'Could not process file',
  training: 'Training failed',
  scoring: 'Scoring failed',
  internal: 'Unexpected error',
};

/**
 * Resolve a user-facing heading + message from the structured job error.
 *
 * Priority:
 *   1. Look up the per-code copy if `errorCode` matches a known value.
 *   2. Otherwise fall back to the per-type heading.
 *   3. As a last resort use a generic heading.
 *
 * The returned `message` prefers the server-provided `error` string (which
 * the worker writes as a clean user-facing message) and falls back to the
 * per-code default if the server did not provide one.
 */
export function resolveJobError(job: {
  error?: string;
  errorCode?: string;
  errorType?: string;
}): { heading: string; message: string } {
  const copy = job.errorCode ? ERROR_COPY[job.errorCode] : undefined;
  const typeHeading = job.errorType
    ? TYPE_HEADINGS[job.errorType as JobErrorType]
    : undefined;

  const heading = copy?.heading ?? typeHeading ?? 'Job failed';
  const message =
    job.error && job.error.trim().length > 0
      ? job.error
      : copy?.fallbackMessage ?? 'An unknown error occurred.';

  return { heading, message };
}
