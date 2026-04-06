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

describe('vertexAi service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCancelCustomJob.mockResolvedValue(undefined);
  });

  it('Test 1: cancelVertexAIJob with valid job resource name returns success', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { JobServiceClient } = await import('@google-cloud/aiplatform');

    await cancelVertexAIJob('test-job-123');

    expect(JobServiceClient).toHaveBeenCalled();
    expect(mockCancelCustomJob).toHaveBeenCalledWith({
      name: expect.stringContaining('test-job-123'),
    });
  });

  it('Test 2: cancelVertexAIJob with non-existent job logs warning (no throw)', async () => {
    mockCancelCustomJob.mockRejectedValueOnce(new Error('Job not found'));
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // Should not throw
    await expect(cancelVertexAIJob('nonexistent-job')).resolves.toBeUndefined();

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
    await expect(cancelVertexAIJob('error-job')).resolves.toBeUndefined();

    // Should log warning
    expect(mockLogger.warn).toHaveBeenCalled();
  });

  it('Test 4: cancelVertexAIJob constructs correct JobServiceClient with location endpoint', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { JobServiceClient } = await import('@google-cloud/aiplatform');

    await cancelVertexAIJob('test-job');

    // Verify client constructed with correct endpoint
    expect(JobServiceClient).toHaveBeenCalledWith({
      apiEndpoint: expect.stringContaining('-aiplatform.googleapis.com'),
    });
  });

  it('Test 5: Mock verifies client.cancelCustomJob called with correct resource name', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    const jobId = 'my-job-456';
    await cancelVertexAIJob(jobId);

    expect(mockCancelCustomJob).toHaveBeenCalledWith({
      name: expect.stringMatching(/projects\/.*\/locations\/.*\/customJobs\/my-job-456/),
    });
  });
});
