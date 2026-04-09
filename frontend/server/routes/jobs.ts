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

// 1. Get Signed URL for Upload
router.post("/upload-url", requireAuth, uploadUrlLimiter, validateUploadUrl, async (req, res) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();

    // Use safe filename - discard user-provided filename for storage path
    const gcsFileName = generateSafeFilename((req as any).user.id);

    const [url] = await storage
      .bucket(BUCKET_NAME)
      .file(gcsFileName)
      .getSignedUrl({
        version: "v4",
        action: "write",
        expires: Date.now() + 15 * 60 * 1000, // 15 minutes
        contentType: contentType || 'text/csv',
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
router.post("/start-job", requireAuth, uploadLimiter, validateStartJob, async (req, res) => {
  try {
    const { jobId, gcsFileName } = req.body;
    const userId = (req as any).user.id;

    // Authorization: ensure the caller owns the file path. Signed URLs are
    // issued under `uploads/<userId>/...` (see generateSafeFilename), so any
    // gcsFileName outside that prefix was not minted for this user. Without
    // this check, an authenticated user who learned another user's object
    // path could trigger processing on files outside their namespace.
    const expectedPrefix = `uploads/${userId}/`;
    if (typeof gcsFileName !== 'string' || !gcsFileName.startsWith(expectedPrefix)) {
      logger.warn("start-job rejected: gcsFileName outside caller namespace", {
        jobId,
        userId,
        gcsFileName,
      });
      return res.status(403).json({ error: "Forbidden: file path does not belong to caller" });
    }

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
      userId,
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
    const userId = (req as any).user.id;
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ status: "not_found" });
    }

    const data = doc.data();

    // Authorization: only the job owner can read its status/results.
    // Return 404 (not 403) so callers cannot use the response code to confirm
    // existence of someone else's job UUIDs.
    if (!data || data.userId !== userId) {
      logger.warn("job-status rejected: caller is not job owner", {
        jobId: id,
        userId,
        ownerId: data?.userId,
      });
      return res.status(404).json({ status: "not_found" });
    }

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
