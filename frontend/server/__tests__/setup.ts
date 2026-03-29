/**
 * Jest test setup
 * Runs before each test file
 */

// Set test environment variables for envalid validation
process.env.NODE_ENV = 'test';
process.env.FRONTEND_URL = 'http://localhost:5173';
process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
process.env.GCS_BUCKET_NAME = 'test-bucket';
process.env.SESSION_SECRET = 'test-secret-key-minimum-32-characters-long';

// NOTE: Google Cloud client mocks should be placed in individual test files
// using jest.mock() at the top of each test file, as needed.
