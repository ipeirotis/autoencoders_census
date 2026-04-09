/**
 * Jobs API Routes - Handles the 3-step upload workflow:
 *
 * 1. POST  /upload-url      - Generates a signed GCS URL for direct browser upload
 * 2. POST  /start-job       - Creates Firestore doc + publishes Pub/Sub message to trigger worker
 * 3. GET   /job-status/:id  - Polls Firestore for job status and results
 * 4. DELETE /:id            - Marks a job as canceled in Firestore
 *
 * Flow: Frontend → (signed URL) → GCS → Pub/Sub → worker.py → Vertex AI → Firestore → Frontend polls
 */

import { Router, Request, Response } from "express";
import { GetSignedUrlConfig } from "@google-cloud/storage";
import { Timestamp } from "@google-cloud/firestore";
import { v4 as uuidv4 } from "uuid";
import { requireAuth } from '../middleware/auth';
import { uploadLimiter, uploadUrlLimiter, pollLimiter, downloadLimiter } from '../middleware/rateLimits';
import { validateJobId, validateUploadUrl, validateStartJob } from '../middleware/validation';
import { generateSafeFilename } from '../utils/fileValidation';
import { logger } from '../config/logger';
import { env } from '../config/env';
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

// Configuration from validated environment variables (see config/env.ts).
// Using the validated `env` object instead of `process.env` directly so a
// missing/misspelled topic or bucket fails fast at startup rather than at
// first request with an opaque "topic not found" runtime error.
const BUCKET_NAME = env.GCS_BUCKET_NAME;
const TOPIC_ID = env.PUBSUB_TOPIC_ID;

/**
 * WORK-12: Defense-in-depth CSV validation
 *
 * Express layer (quick checks):
 *   - File extension validation (.csv only) - handled by validateUploadUrl middleware
 *   - Content-Type passthrough so the signed URL's Content-Type constraint
 *     matches the client's subsequent PUT (see /upload-url handler).
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
router.post("/upload-url", requireAuth, uploadUrlLimiter, validateUploadUrl, async (req: Request, res: Response) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();

    // Use safe filename - discard user-provided filename for storage path
    const gcsFileName = generateSafeFilename((req as any).user.id);

    // V4 signed URLs include Content-Type in the canonical request, so the
    // client's PUT must send the exact same value the URL was signed for. If
    // the client did not declare a contentType (e.g. browsers that report an
    // empty MIME for `.csv` files), omit it from the signing options entirely
    // so the signature does not constrain the Content-Type header. Otherwise
    // sign for the exact value the client said it would send so they match -
    // including Excel-era MIME types like `application/vnd.ms-excel` that
    // Windows browsers routinely report for CSV files.
    const signOptions: GetSignedUrlConfig = {
      version: "v4",
      action: "write",
      expires: Date.now() + 15 * 60 * 1000, // 15 minutes
    };
    if (typeof contentType === 'string' && contentType.length > 0) {
      signOptions.contentType = contentType;
    }

    const [url] = await storage
      .bucket(BUCKET_NAME)
      .file(gcsFileName)
      .getSignedUrl(signOptions);

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

    // Atomically claim the job ID. Use Firestore's `create()` (not `set()`):
    // it throws ALREADY_EXISTS if the doc is taken, which prevents an
    // authenticated user who learned another user's job UUID (via logs,
    // shared client state, etc.) from overwriting `userId`/status on the
    // existing doc and locking the original owner out of /job-status/:id.
    //
    // The persisted status MUST be "queued" (JobStatus.QUEUED in worker.py),
    // not "uploading". The worker's is_valid_transition(None, ...) state
    // machine only accepts QUEUED as the valid first status - any other
    // value would cause update_job_status(..., PROCESSING) to raise
    // ValueError and leave the job stuck forever without ever running.
    //
    // Codex P1 (r3053917215 + follow-up r3053955xxx): publish failure
    // handling is trickier than just "delete on publish error". If
    // publishMessage() actually succeeded on the server side but the
    // client saw a transient network error (timeout, reset), Pub/Sub
    // still delivers the message and the worker needs the claimed doc
    // to exist - deleting it would strand a valid job. Conversely, if
    // publish truly failed we do want the client to be able to retry
    // with the same jobId (otherwise the 409 guard locks them out).
    // Squaring this: keep the doc on publish error, and make a
    // same-user retry with the same jobId idempotent - if the existing
    // claim is still "queued" and owned by the caller, reuse it and
    // re-publish. The worker-side dedup (check_idempotency and
    // mark_message_processed in worker.py) makes a duplicate publish
    // on retry harmless.
    let claimedExistingDoc = false;
    try {
      await firestore.collection("jobs").doc(jobId).create({
        status: "queued", // Initial state (matches worker JobStatus.QUEUED)
        createdAt: new Date(),
        userId,
      });
    } catch (createErr: any) {
      // Firestore raises code 6 (ALREADY_EXISTS) when create() hits an
      // existing document.
      if (createErr?.code !== 6) {
        throw createErr;
      }

      // Same-user retry recovery: if the existing doc is owned by this
      // caller and still in the initial "queued" state (no worker has
      // picked it up yet), treat the request as a publish retry and
      // reuse the claim. Any other combination (different owner, or a
      // status past "queued") is a real collision / replay attempt and
      // gets the 409.
      const existingDoc = await firestore.collection("jobs").doc(jobId).get();
      const existing = existingDoc.exists ? existingDoc.data() : undefined;
      if (
        existing &&
        existing.userId === userId &&
        existing.status === "queued"
      ) {
        claimedExistingDoc = true;
        logger.info("start-job: reusing pre-existing queued doc for retry", {
          jobId,
          userId,
        });
      } else {
        logger.warn("start-job rejected: jobId already claimed", {
          jobId,
          userId,
          existingOwner: existing?.userId,
          existingStatus: existing?.status,
        });
        return res.status(409).json({ error: "Job ID already exists" });
      }
    }

    // Publish to Pub/Sub. On failure we do NOT delete the doc: the publish
    // may have succeeded server-side even though the client saw an error,
    // and we must not strand a valid job just because the ack was lost in
    // flight. The client is expected to retry /start-job with the same
    // jobId on any failure; the same-user reuse branch above will see the
    // still-"queued" doc and re-publish. The worker's Pub/Sub-level dedup
    // prevents duplicate processing if the first publish actually landed.
    await pubsub.topic(TOPIC_ID).publishMessage({ data: dataBuffer });

    res.json({
      success: true,
      message: claimedExistingDoc ? "Job re-enqueued" : "Job started",
    });
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

    // Normalize Firestore Timestamps to ISO strings so clients can
    // construct valid Date objects from the response.
    res.json(serializeFirestoreData(data));
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
    const userId = (req as any).user.id;
    const docRef = firestore.collection("jobs").doc(id);
    const doc = await docRef.get();

    if (!doc.exists) {
      // Return 404 so callers cannot probe for other users' job UUIDs.
      return res.status(404).json({ error: "Job not found" });
    }

    const data = doc.data();

    // Authorization: only the job owner can cancel. Match the /job-status
    // convention and return 404 (not 403) to avoid confirming UUID existence.
    if (!data || data.userId !== userId) {
      logger.warn("cancel-job rejected: caller is not job owner", {
        jobId: id,
        userId,
        ownerId: data?.userId,
      });
      return res.status(404).json({ error: "Job not found" });
    }

    // Don't re-cancel jobs that already reached a terminal state
    const terminalStates = new Set(["complete", "error", "canceled"]);
    if (data.status && terminalStates.has(data.status)) {
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
