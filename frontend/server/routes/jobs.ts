/**
 * Jobs API Routes - Handles the 3-step upload workflow:
 *
 * 1. POST /upload-url       - Generates a signed GCS URL for direct browser upload
 * 2. POST /start-job        - Creates Firestore doc + publishes Pub/Sub message to trigger worker
 * 3. GET  /job-status/:id   - Polls Firestore for job status and results
 * 4. DELETE /:id            - Marks a job as canceled in Firestore
 *
 * Flow: Frontend → (signed URL) → GCS → Pub/Sub → worker.py → Vertex AI → Firestore → Frontend polls
 */

import { Router, Request, Response } from "express";
import { Timestamp } from "@google-cloud/firestore";
import { v4 as uuidv4 } from "uuid";
import { requireAuth } from '../middleware/auth';
import { uploadLimiter, pollLimiter, downloadLimiter } from '../middleware/rateLimits';
import { validateJobId, validateUploadUrl, validateStartJob } from '../middleware/validation';
import { generateSafeFilename } from '../utils/fileValidation';
import { logger } from '../config/logger';
import { storage, firestore, pubsub } from '../config/gcp-clients';

/**
 * Recursively convert Firestore Timestamp values into ISO strings so that
 * clients receive parseable date values instead of {_seconds, _nanoseconds}
 * objects. Plain values (strings, numbers, booleans) are returned unchanged.
 */
function serializeFirestoreData(value: unknown): unknown {
  if (value instanceof Timestamp) {
    return value.toDate().toISOString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  // Duck-type fallback for Timestamp-shaped objects (defensive against SDK
  // variants that may produce plain objects with a toDate() method).
  if (
    value &&
    typeof value === 'object' &&
    typeof (value as { toDate?: unknown }).toDate === 'function'
  ) {
    const date = (value as { toDate: () => Date }).toDate();
    if (date instanceof Date && !Number.isNaN(date.getTime())) {
      return date.toISOString();
    }
  }
  if (Array.isArray(value)) {
    return value.map(serializeFirestoreData);
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, serializeFirestoreData(v)])
    );
  }
  return value;
}

const router = Router();

// Configuration from Environment Variables
const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "your-bucket-name";
const TOPIC_ID = process.env.PUBSUB_TOPIC_ID || "your-topic-id";

// 1. Get Signed URL for Upload
// WORK-12: Defense-in-depth CSV validation
// Express layer (quick checks):
//   - File extension validation (.csv only) - handled by validateUploadUrl middleware
//   - Content-Type validation (warn on unexpected types, allow for browser compatibility)
// Worker layer (deep checks):
//   - Encoding detection and validation (chardet)
//   - Structure validation (pandas streaming, min rows/cols)
//   - Size limit enforcement (>100MB rejection)
//   - Edge case handling (unicode, missing values)
// Note: File size validation (WORK-11) happens at Worker layer via validate_csv()
// Express layer cannot reliably check size before GCS upload completes
// GCS bucket has 100MB object size limit configured separately
router.post("/upload-url", requireAuth, uploadLimiter, validateUploadUrl, async (req: Request, res: Response) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();

    // WORK-12: Content-Type validation (defensive, not strict)
    // Some browsers send different MIME types for CSV files
    if (contentType && contentType !== 'text/csv' && contentType !== 'application/vnd.ms-excel') {
      logger.warn('Unexpected content type for CSV upload', {
        contentType,
        filename,
        userId: (req as any).user.id
      });
      // Allow but log - browser MIME type detection is inconsistent
    }

    // Use safe filename - discard user-provided filename for storage path
    const gcsFileName = generateSafeFilename((req as any).user.id);

    const [url] = await storage
      .bucket(BUCKET_NAME)
      .file(gcsFileName)
      .getSignedUrl({
        version: "v4",
        action: "write",
        expires: Date.now() + 15 * 60 * 1000, // 15 minutes
        contentType: 'text/csv',  // Force CSV content type for GCS
      });

    res.json({ url, jobId, gcsFileName, originalFilename: filename });
  } catch (error) {
    logger.error("Error generating signed URL", {
      error: error instanceof Error ? error.message : String(error),
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to generate upload URL" });
  }
});

// 2. Start Processing (Trigger Pub/Sub)
router.post("/start-job", requireAuth, uploadLimiter, validateStartJob, async (req: Request, res: Response) => {
  try {
    const { jobId, gcsFileName } = req.body;

    // The message format MUST match what worker.py expects
    const messageJson = {
      jobId: jobId,
      bucket: BUCKET_NAME,
      file: gcsFileName,
    };

    const dataBuffer = Buffer.from(JSON.stringify(messageJson));
    await pubsub.topic(TOPIC_ID).publishMessage({ data: dataBuffer });

    // Initialize Firestore document
    await firestore.collection("jobs").doc(jobId).set({
      status: "uploading", // Initial state
      createdAt: new Date(),
      userId: (req as any).user.id,
    });

    res.json({ success: true, message: "Job started" });
  } catch (error) {
    logger.error("Error starting job", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.body.jobId,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to start job" });
  }
});

// 3. Check Job Status
router.get("/job-status/:id", requireAuth, pollLimiter, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ status: "not_found" });
    }

    // Normalize Firestore Timestamps to ISO strings so clients can
    // construct valid Date objects from the response.
    res.json(serializeFirestoreData(doc.data()));
  } catch (error) {
    logger.error("Error checking status", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.params.id,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to check status" });
  }
});

// 4. Cancel Job
// Marks the Firestore job document as "canceled" so the polling client
// reaches a terminal state and stops polling. The worker is expected to
// observe this state change and abort any in-flight processing.
router.delete("/:id", requireAuth, pollLimiter, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const docRef = firestore.collection("jobs").doc(id);
    const doc = await docRef.get();

    if (!doc.exists) {
      return res.status(404).json({ error: "Job not found" });
    }

    const data = doc.data();
    if (data?.userId && data.userId !== (req as any).user.id) {
      return res.status(404).json({ error: "Job not found" });
    }

    // Don't re-cancel jobs that already reached a terminal state
    const terminalStates = new Set(["complete", "error", "canceled"]);
    if (data?.status && terminalStates.has(data.status)) {
      return res.status(409).json({
        error: "Job has already reached a terminal state",
        status: data.status,
      });
    }

    await docRef.update({
      status: "canceled",
      canceledAt: new Date(),
      updatedAt: new Date(),
    });

    res.json({ success: true, jobId: id, status: "canceled" });
  } catch (error) {
    logger.error("Error canceling job", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.params.id,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to cancel job" });
  }
});

export const jobsRouter = router;