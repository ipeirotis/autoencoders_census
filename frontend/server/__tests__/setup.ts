/**
 * Jest test setup
 * Runs before each test file
 * Sets test environment variables for validation
 */

// Set test environment variables for envalid validation
process.env.NODE_ENV = 'test';
process.env.FRONTEND_URL = 'http://localhost:5173';
process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
process.env.GCS_BUCKET_NAME = 'test-bucket';
process.env.PUBSUB_TOPIC_ID = 'test-topic';
process.env.SESSION_SECRET = 'test-secret-key-minimum-32-characters-long';

// Note: Google Cloud client mocks (Firestore, Storage, PubSub) should be
// placed at the top of individual test files using jest.mock() as needed.
