/**
 * Jobs API Routes - Handles the 3-step upload workflow:
 *
 * 1. POST /upload-url  - Generates a signed GCS URL for direct browser upload
 * 2. POST /start-job   - Creates Firestore doc + publishes Pub/Sub message to trigger worker
 * 3. GET /job-status/:id - Polls Firestore for job status and results
 *
 * Flow: Frontend → (signed URL) → GCS → Pub/Sub → worker.py → Vertex AI → Firestore → Frontend polls
 */

import { Router } from "express";
import { Storage } from "@google-cloud/storage";
import { Firestore } from "@google-cloud/firestore";
import { PubSub } from "@google-cloud/pubsub";
import { v4 as uuidv4 } from "uuid";
import { requireAuth } from '../middleware/auth';
import { uploadLimiter, uploadUrlLimiter, pollLimiter, downloadLimiter } from '../middleware/rateLimits';
import { validateJobId, validateUploadUrl, validateStartJob } from '../middleware/validation';
import { generateSafeFilename } from '../utils/fileValidation';
import { logger } from '../config/logger';

const router = Router();
const storage = new Storage();
const firestore = new Firestore();
const pubsub = new PubSub();

// Configuration from Environment Variables
const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "your-bucket-name";
const TOPIC_ID = process.env.PUBSUB_TOPIC_ID || "your-topic-id";

// Accepted CSV content types. Browsers (especially on Windows) can label
// CSV files as application/vnd.ms-excel instead of text/csv, so we accept
// both and fall back to text/csv for anything else.
const ALLOWED_CSV_CONTENT_TYPES = ['text/csv', 'application/vnd.ms-excel'];
const DEFAULT_CSV_CONTENT_TYPE = 'text/csv';

/**
 * WORK-12: Defense-in-depth CSV validation
 *
 * Express layer (quick checks):
 *   - File extension validation (.csv only) - handled by validateUploadUrl middleware
 *   - Content-Type validation (warn on unexpected types, allow for browser compatibility)
 *
 * Worker layer (deep checks):
 *   - Encoding detection and validation (chardet)
 *   - Structure validation (pandas streaming, min rows/cols)
 *   - Size limit enforcement (>100MB rejection)
 *   - Edge case handling (unicode, missing values)
 *
 * Note: File size validation (WORK-11) happens at the Worker layer via
 * validate_csv(); the Express layer cannot reliably check size before
 * the GCS upload completes. The GCS bucket has a 100MB object size
 * limit configured separately.
 *
 * /upload-url uses the dedicated uploadUrlLimiter (separate from the
 * uploadLimiter used by /start-job) so a normal 3-step upload doesn't
 * consume two rate-limit slots from the same budget - see
 * frontend/server/middleware/rateLimits.ts.
 */

// 1. Get Signed URL for Upload
router.post("/upload-url", requireAuth, uploadUrlLimiter, validateUploadUrl, async (req, res) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();

    // WORK-12: Content-Type validation (defensive, not strict)
    // Some browsers send different MIME types for CSV files. Use the client's
    // content type when it is on the allow list so the signed URL accepts the
    // subsequent PUT (GCS enforces an exact Content-Type match when one was
    // specified at signing time). For unexpected values, warn and fall back
    // to text/csv so the URL still works for standard CSV uploads.
    let signedUrlContentType = DEFAULT_CSV_CONTENT_TYPE;
    if (typeof contentType === 'string' && contentType.length > 0) {
      if (ALLOWED_CSV_CONTENT_TYPES.includes(contentType)) {
        signedUrlContentType = contentType;
      } else {
        logger.warn('Unexpected content type for CSV upload', {
          contentType,
          filename,
          userId: (req as any).user.id
        });
        // Fall through using DEFAULT_CSV_CONTENT_TYPE
      }
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
        contentType: signedUrlContentType,
      });

    res.json({ url, jobId, gcsFileName, originalFilename: filename, contentType: signedUrlContentType });
  } catch (error) {
    logger.error("Error generating signed URL", {
      error: error instanceof Error ? error.message : String(error),
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to generate upload URL" });
  }
});

// 2. Start Processing (Trigger Pub/Sub)
router.post("/start-job", requireAuth, uploadLimiter, validateStartJob, async (req, res) => {
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
    // Use "queued" to match the worker's JobStatus state machine (worker.py):
    // is_valid_transition(None, QUEUED) is True, and QUEUED is the only valid
    // first status. Using any other string (e.g. "uploading") would cause the
    // worker's first update_job_status() call to raise ValueError and leave
    // the job stuck without ever transitioning to processing.
    await firestore.collection("jobs").doc(jobId).set({
      status: "queued", // Initial state (matches worker JobStatus.QUEUED)
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
router.get("/job-status/:id", requireAuth, pollLimiter, validateJobId, async (req, res) => {
  try {
    const { id } = req.params;
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ status: "not_found" });
    }

    const data = doc.data();
    res.json(data);
  } catch (error) {
    logger.error("Error checking status", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.params.id,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to check status" });
  }
});

export const jobsRouter = router;
