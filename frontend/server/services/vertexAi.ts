/**
 * Vertex AI service module for job management operations
 */

import { JobServiceClient, PipelineServiceClient } from '@google-cloud/aiplatform';
import { logger } from '../config/logger';
import { env } from '../config/env';

const PROJECT_ID = env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.VERTEX_AI_LOCATION || 'us-central1';

// Accept either flavor of Vertex resource we might have stored:
//   projects/{p}/locations/{l}/trainingPipelines/{id}
//     ^ what CustomContainerTrainingJob.run() yields (what we actually submit)
//   projects/{p}/locations/{l}/customJobs/{id}
//     ^ only if the caller extracts the underlying CustomJob explicitly
const TRAINING_PIPELINE_RE =
  /^projects\/[^/]+\/locations\/[^/]+\/trainingPipelines\/[^/]+$/;
const CUSTOM_JOB_RE =
  /^projects\/[^/]+\/locations\/[^/]+\/customJobs\/[^/]+$/;

/**
 * Cancel a Vertex AI training resource using the actual server-generated
 * resource name persisted by the worker.
 *
 * worker.py submits training via `aiplatform.CustomContainerTrainingJob.run()`
 * which creates a `TrainingPipeline` on the server. The Python SDK exposes
 * that pipeline's resource name via `job.resource_name`, in the form
 *   projects/{p}/locations/{l}/trainingPipelines/{id}
 * The pipeline in turn orchestrates one or more CustomJobs, but their
 * server-generated IDs are not directly available from the wrapper before
 * the job is running. Cancelling the *pipeline* is what actually stops the
 * work, so that's what we do.
 *
 * This helper also accepts raw customJobs/... resource names for callers
 * that have extracted them, for backwards compatibility and flexibility.
 *
 * Cancellation is asynchronous and best-effort. The resource may already be
 * complete or in a non-cancellable state. This function does not throw on
 * failure, so cleanup can continue even when cancellation is not possible.
 *
 * @param resourceName - Full Vertex AI resource name (trainingPipelines/... or
 *   customJobs/...) as stored on the Firestore job document, or null/undefined
 *   if the job was never dispatched to Vertex AI (e.g. local mode).
 * @param appJobId - Application job ID, included in log lines for traceability.
 */
export async function cancelVertexAIJob(
  resourceName: string | null | undefined,
  appJobId?: string
): Promise<boolean> {
  if (!resourceName) {
    logger.info('Skipping Vertex AI cancellation - no resource name stored', {
      appJobId,
    });
    return true; // nothing to cancel = success
  }

  const isTrainingPipeline = TRAINING_PIPELINE_RE.test(resourceName);
  const isCustomJob = CUSTOM_JOB_RE.test(resourceName);

  if (!isTrainingPipeline && !isCustomJob) {
    logger.warn('Refusing to cancel Vertex AI job - not a recognized resource path', {
      appJobId,
      resourceName,
    });
    return false;
  }

  // Derive the regional API endpoint from the location segment embedded in the
  // resource name itself, rather than the global VERTEX_AI_LOCATION env var.
  // The worker submits jobs with a hardcoded location ("us-central1") which may
  // differ from the server's env config; using the wrong region sends the
  // cancel request to the wrong endpoint where it silently no-ops.
  const locationMatch = resourceName.match(/\/locations\/([^/]+)\//);
  const location = locationMatch ? locationMatch[1] : LOCATION;
  const apiEndpoint = `${location}-aiplatform.googleapis.com`;

  try {
    if (isTrainingPipeline) {
      const client = new PipelineServiceClient({ apiEndpoint });
      await client.cancelTrainingPipeline({ name: resourceName });
      logger.info('Vertex AI training pipeline cancellation requested', {
        appJobId,
        resourceName,
      });
    } else {
      const client = new JobServiceClient({ apiEndpoint });
      await client.cancelCustomJob({ name: resourceName });
      logger.info('Vertex AI custom job cancellation requested', {
        appJobId,
        resourceName,
      });
    }
    return true;
  } catch (error) {
    logger.warn('Failed to cancel Vertex AI resource', {
      appJobId,
      resourceName,
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

export const __vertexAiConfig = { PROJECT_ID, LOCATION };
