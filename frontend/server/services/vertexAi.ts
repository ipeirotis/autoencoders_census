/**
 * Vertex AI service module for job management operations
 */

import { JobServiceClient } from '@google-cloud/aiplatform';
import { logger } from '../config/logger';
import { env } from '../config/env';

const PROJECT_ID = env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.VERTEX_AI_LOCATION || 'us-central1';

/**
 * Cancel a Vertex AI CustomJob
 *
 * Note: Cancellation is asynchronous and best-effort. The job may already be
 * complete or in a non-cancellable state. This function does not throw on failure
 * to ensure cleanup continues even if cancellation is not possible.
 *
 * @param jobId - The job ID (used to construct resource name)
 */
export async function cancelVertexAIJob(jobId: string): Promise<void> {
  const client = new JobServiceClient({
    apiEndpoint: `${LOCATION}-aiplatform.googleapis.com`
  });

  // Construct resource name for the CustomJob
  // Format: projects/{project}/locations/{location}/customJobs/{jobId}
  const resourceName = `projects/${PROJECT_ID}/locations/${LOCATION}/customJobs/${jobId}`;

  try {
    await client.cancelCustomJob({ name: resourceName });
    logger.info(`Vertex AI job cancellation requested`, { jobId, resourceName });
  } catch (error) {
    // Cancellation is asynchronous and best-effort (not guaranteed)
    // Job may already be complete or not exist
    logger.warn(`Failed to cancel Vertex AI job`, {
      jobId,
      error: error instanceof Error ? error.message : String(error)
    });
    // Do NOT throw - cancellation is best-effort
  }
}
