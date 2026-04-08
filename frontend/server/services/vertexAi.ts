/**
 * Vertex AI service module for job management operations
 */

import { JobServiceClient } from '@google-cloud/aiplatform';
import { logger } from '../config/logger';
import { env } from '../config/env';

const PROJECT_ID = env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.VERTEX_AI_LOCATION || 'us-central1';

/**
 * Cancel a Vertex AI CustomJob using the actual server-generated resource name.
 *
 * The Vertex AI API requires the full server-side resource name to cancel a job:
 *   projects/{project}/locations/{location}/customJobs/{vertexJobId}
 *
 * The {vertexJobId} segment is a numeric ID assigned by Vertex AI when the
 * CustomJob is created. It is NOT the same as our application's job UUID.
 * The worker stores this resource name in Firestore (`vertexJobName`) when it
 * dispatches the job; cancellation must use that stored value.
 *
 * Cancellation is asynchronous and best-effort. The job may already be
 * complete or in a non-cancellable state. This function does not throw on
 * failure to ensure cleanup continues even if cancellation is not possible.
 *
 * @param vertexJobName - The full Vertex AI resource name as stored on the
 *   Firestore job document, or null/undefined if the job was never dispatched
 *   to Vertex AI (e.g. local mode).
 * @param appJobId - Application job ID, included in log lines for traceability.
 */
export async function cancelVertexAIJob(
  vertexJobName: string | null | undefined,
  appJobId?: string
): Promise<void> {
  if (!vertexJobName) {
    // Job was never dispatched to Vertex AI (e.g. local mode, or canceled
    // before the worker submitted the CustomJob). Nothing to cancel.
    logger.info('Skipping Vertex AI cancellation - no resource name stored', {
      appJobId,
    });
    return;
  }

  // Sanity check the resource name format. Reject anything that does not look
  // like a Vertex resource path - this catches legacy docs where we may have
  // mistakenly stored an app UUID instead of the real resource name.
  const isResourceName = /^projects\/[^/]+\/locations\/[^/]+\/customJobs\/[^/]+$/.test(
    vertexJobName
  );
  if (!isResourceName) {
    logger.warn('Refusing to cancel Vertex AI job - vertexJobName is not a resource path', {
      appJobId,
      vertexJobName,
    });
    return;
  }

  const client = new JobServiceClient({
    apiEndpoint: `${LOCATION}-aiplatform.googleapis.com`,
  });

  try {
    await client.cancelCustomJob({ name: vertexJobName });
    logger.info('Vertex AI job cancellation requested', {
      appJobId,
      vertexJobName,
    });
  } catch (error) {
    // Cancellation is asynchronous and best-effort (not guaranteed)
    // Job may already be complete or not exist
    logger.warn('Failed to cancel Vertex AI job', {
      appJobId,
      vertexJobName,
      error: error instanceof Error ? error.message : String(error),
    });
    // Do NOT throw - cancellation is best-effort
  }
}

// Re-export resolved config for tests / diagnostics
export const __vertexAiConfig = { PROJECT_ID, LOCATION };
