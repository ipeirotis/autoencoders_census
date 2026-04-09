/**
 * Singleton GCP client instances
 *
 * Creates single instances of Storage, Firestore, and PubSub clients
 * to prevent connection pool exhaustion from multiple instantiations.
 * All routes should import from this module instead of creating new clients.
 */

import { Storage } from '@google-cloud/storage';
import { Firestore } from '@google-cloud/firestore';
import { PubSub } from '@google-cloud/pubsub';
import { env } from './env';

/**
 * Singleton Storage client
 * Used for GCS bucket operations (file uploads, signed URLs)
 */
export const storage = new Storage({ projectId: env.GOOGLE_CLOUD_PROJECT });

/**
 * Singleton Firestore client
 * Used for job metadata storage and status tracking
 */
export const firestore = new Firestore({ projectId: env.GOOGLE_CLOUD_PROJECT });

/**
 * Singleton PubSub client
 * Used for publishing messages to trigger worker processing
 */
export const pubsub = new PubSub({ projectId: env.GOOGLE_CLOUD_PROJECT });
