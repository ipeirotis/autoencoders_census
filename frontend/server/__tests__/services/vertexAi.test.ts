/**
 * Tests for Vertex AI service module
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';

// Mock the JobServiceClient
const mockCancelCustomJob = jest.fn().mockResolvedValue(undefined);
jest.unstable_mockModule('@google-cloud/aiplatform', () => ({
  JobServiceClient: jest.fn().mockImplementation(() => ({
    cancelCustomJob: mockCancelCustomJob,
  })),
}));

// Mock logger
const mockLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};
jest.unstable_mockModule('../../config/logger', () => ({
  logger: mockLogger,
}));

const VALID_RESOURCE_NAME =
  'projects/test-project/locations/us-central1/customJobs/1234567890';

describe('vertexAi service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCancelCustomJob.mockResolvedValue(undefined);
  });

  it('Test 1: cancelVertexAIJob with a valid resource name calls cancelCustomJob', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { JobServiceClient } = await import('@google-cloud/aiplatform');

    await cancelVertexAIJob(VALID_RESOURCE_NAME, 'app-job-1');

    expect(JobServiceClient).toHaveBeenCalled();
    expect(mockCancelCustomJob).toHaveBeenCalledWith({
      name: VALID_RESOURCE_NAME,
    });
  });

  it('Test 2: cancelVertexAIJob with non-existent job logs warning (no throw)', async () => {
    mockCancelCustomJob.mockRejectedValueOnce(new Error('Job not found'));
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // Should not throw
    await expect(
      cancelVertexAIJob(VALID_RESOURCE_NAME, 'app-job-2')
    ).resolves.toBeUndefined();

    // Should log warning
    expect(mockLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Failed to cancel'),
      expect.any(Object)
    );
  });

  it('Test 3: cancelVertexAIJob with API error logs warning (no throw)', async () => {
    mockCancelCustomJob.mockRejectedValueOnce(new Error('API Error'));
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // Should not throw
    await expect(
      cancelVertexAIJob(VALID_RESOURCE_NAME, 'app-job-3')
    ).resolves.toBeUndefined();

    // Should log warning
    expect(mockLogger.warn).toHaveBeenCalled();
  });

  it('Test 4: cancelVertexAIJob constructs JobServiceClient with location endpoint', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { JobServiceClient } = await import('@google-cloud/aiplatform');

    await cancelVertexAIJob(VALID_RESOURCE_NAME);

    // Verify client constructed with correct endpoint
    expect(JobServiceClient).toHaveBeenCalledWith({
      apiEndpoint: expect.stringContaining('-aiplatform.googleapis.com'),
    });
  });

  it('Test 5: cancelVertexAIJob passes the exact stored resource name through', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    const resourceName =
      'projects/test-project/locations/us-central1/customJobs/9876543210';
    await cancelVertexAIJob(resourceName);

    expect(mockCancelCustomJob).toHaveBeenCalledWith({
      name: resourceName,
    });
    expect(mockCancelCustomJob).toHaveBeenCalledWith({
      name: expect.stringMatching(
        /^projects\/.+\/locations\/.+\/customJobs\/9876543210$/
      ),
    });
  });

  it('Test 6: cancelVertexAIJob is a no-op when vertexJobName is missing', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    await cancelVertexAIJob(undefined, 'app-job-no-vertex');
    await cancelVertexAIJob(null, 'app-job-no-vertex');
    await cancelVertexAIJob('', 'app-job-no-vertex');

    // Should never invoke the underlying client when there is nothing to cancel
    expect(mockCancelCustomJob).not.toHaveBeenCalled();
  });

  it('Test 7: cancelVertexAIJob refuses to call API with a non-resource-path value', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // An app UUID is NOT a valid Vertex resource path; we must not call the API
    await cancelVertexAIJob('an-app-uuid-1234', 'app-job-bad-name');

    expect(mockCancelCustomJob).not.toHaveBeenCalled();
    expect(mockLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Refusing to cancel'),
      expect.any(Object)
    );
  });
});
