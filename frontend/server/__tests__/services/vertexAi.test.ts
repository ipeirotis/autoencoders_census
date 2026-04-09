/**
 * Tests for Vertex AI service module
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';

// Mock both clients. cancelVertexAIJob picks one based on the resource name.
const mockCancelCustomJob = jest.fn().mockResolvedValue(undefined);
const mockCancelTrainingPipeline = jest.fn().mockResolvedValue(undefined);
jest.unstable_mockModule('@google-cloud/aiplatform', () => ({
  JobServiceClient: jest.fn().mockImplementation(() => ({
    cancelCustomJob: mockCancelCustomJob,
  })),
  PipelineServiceClient: jest.fn().mockImplementation(() => ({
    cancelTrainingPipeline: mockCancelTrainingPipeline,
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

const VALID_CUSTOM_JOB =
  'projects/test-project/locations/us-central1/customJobs/1234567890';
const VALID_TRAINING_PIPELINE =
  'projects/test-project/locations/us-central1/trainingPipelines/9876543210';

describe('vertexAi service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCancelCustomJob.mockResolvedValue(undefined);
    mockCancelTrainingPipeline.mockResolvedValue(undefined);
  });

  it('Test 1: cancelVertexAIJob on a customJobs/... path calls cancelCustomJob', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { JobServiceClient } = await import('@google-cloud/aiplatform');

    await cancelVertexAIJob(VALID_CUSTOM_JOB, 'app-job-1');

    expect(JobServiceClient).toHaveBeenCalled();
    expect(mockCancelCustomJob).toHaveBeenCalledWith({
      name: VALID_CUSTOM_JOB,
    });
    expect(mockCancelTrainingPipeline).not.toHaveBeenCalled();
  });

  it('Test 2: cancelVertexAIJob on a trainingPipelines/... path calls cancelTrainingPipeline', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { PipelineServiceClient } = await import('@google-cloud/aiplatform');

    await cancelVertexAIJob(VALID_TRAINING_PIPELINE, 'app-job-pipeline');

    expect(PipelineServiceClient).toHaveBeenCalled();
    expect(mockCancelTrainingPipeline).toHaveBeenCalledWith({
      name: VALID_TRAINING_PIPELINE,
    });
    expect(mockCancelCustomJob).not.toHaveBeenCalled();
  });

  it('Test 3: cancelVertexAIJob with non-existent job logs warning (no throw)', async () => {
    mockCancelTrainingPipeline.mockRejectedValueOnce(new Error('Not found'));
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // Should not throw
    await expect(
      cancelVertexAIJob(VALID_TRAINING_PIPELINE, 'app-job-2')
    ).resolves.toBeUndefined();

    // Should log warning
    expect(mockLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Failed to cancel'),
      expect.any(Object)
    );
  });

  it('Test 4: cancelVertexAIJob with API error on customJobs logs warning (no throw)', async () => {
    mockCancelCustomJob.mockRejectedValueOnce(new Error('API Error'));
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // Should not throw
    await expect(
      cancelVertexAIJob(VALID_CUSTOM_JOB, 'app-job-3')
    ).resolves.toBeUndefined();

    // Should log warning
    expect(mockLogger.warn).toHaveBeenCalled();
  });

  it('Test 5: cancelVertexAIJob constructs clients with location endpoint', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');
    const { JobServiceClient, PipelineServiceClient } = await import(
      '@google-cloud/aiplatform'
    );

    await cancelVertexAIJob(VALID_CUSTOM_JOB);
    expect(JobServiceClient).toHaveBeenCalledWith({
      apiEndpoint: expect.stringContaining('-aiplatform.googleapis.com'),
    });

    await cancelVertexAIJob(VALID_TRAINING_PIPELINE);
    expect(PipelineServiceClient).toHaveBeenCalledWith({
      apiEndpoint: expect.stringContaining('-aiplatform.googleapis.com'),
    });
  });

  it('Test 6: cancelVertexAIJob passes the exact stored resource name through', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    const pipelineName =
      'projects/test-project/locations/us-central1/trainingPipelines/9876543210';
    await cancelVertexAIJob(pipelineName);

    expect(mockCancelTrainingPipeline).toHaveBeenCalledWith({
      name: pipelineName,
    });
    expect(mockCancelTrainingPipeline).toHaveBeenCalledWith({
      name: expect.stringMatching(
        /^projects\/.+\/locations\/.+\/trainingPipelines\/9876543210$/
      ),
    });
  });

  it('Test 7: cancelVertexAIJob is a no-op when vertexJobName is missing', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    await cancelVertexAIJob(undefined, 'app-job-no-vertex');
    await cancelVertexAIJob(null, 'app-job-no-vertex');
    await cancelVertexAIJob('', 'app-job-no-vertex');

    // Should never invoke the underlying clients when there is nothing to cancel
    expect(mockCancelCustomJob).not.toHaveBeenCalled();
    expect(mockCancelTrainingPipeline).not.toHaveBeenCalled();
  });

  it('Test 8: cancelVertexAIJob refuses to call API with a non-resource-path value', async () => {
    const { cancelVertexAIJob } = await import('../../services/vertexAi');

    // An app UUID is NOT a valid Vertex resource path; we must not call the API
    await cancelVertexAIJob('an-app-uuid-1234', 'app-job-bad-name');

    expect(mockCancelCustomJob).not.toHaveBeenCalled();
    expect(mockCancelTrainingPipeline).not.toHaveBeenCalled();
    expect(mockLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Refusing to cancel'),
      expect.any(Object)
    );
  });
});
