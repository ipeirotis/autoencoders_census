/**
 * Jobs API Routes - Handles the 3-step upload workflow:
 *
 * 1. POST /upload-url  - Generates a signed GCS URL for direct browser upload
 * 2. POST /start-job   - Creates Firestore doc + publishes Pub/Sub message to trigger worker
 * 3. GET /job-status/:id - Polls Firestore for job status and results
 *
 * Phase 4 additions:
 * 4. GET /:id/export    - CSV export with formula injection protection
 * 5. DELETE /:id        - Job cancellation with full resource cleanup
 * 6. DELETE /:id/files  - Manual file deletion for completed jobs
 *
 * Flow: Frontend → (signed URL) → GCS → Pub/Sub → worker.py → Vertex AI → Firestore → Frontend polls
 */

import { Router } from "express";
import { GetSignedUrlConfig } from "@google-cloud/storage";
import { v4 as uuidv4 } from "uuid";
import { format } from 'fast-csv';
import { requireAuth } from '../middleware/auth';
import { uploadLimiter, uploadUrlLimiter, pollLimiter, downloadLimiter } from '../middleware/rateLimits';
import { validateJobId, validateUploadUrl, validateStartJob } from '../middleware/validation';
import { generateSafeFilename } from '../utils/fileValidation';
import { sanitizeFormulaInjection } from '../utils/csvSanitization';
import { logger } from '../config/logger';
import { env } from '../config/env';
import { storage, firestore, pubsub } from '../config/gcp-clients';
import { cancelVertexAIJob } from '../services/vertexAi';

/**
 * Returns true when a GCS delete error means "object already gone".
 * Treating already-deleted as success keeps cleanup idempotent.
 */
function isGcsNotFoundError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const e = error as { code?: unknown };
  return e.code === 404 || e.code === 'ENOENT';
}

const router = Router();

// Configuration from validated environment variables (see config/env.ts).
const BUCKET_NAME = env.GCS_BUCKET_NAME;
const TOPIC_ID = env.PUBSUB_TOPIC_ID;

/**
 * Non-terminal job statuses that a same-owner start-job retry is allowed to
 * resume. Terminal statuses (complete, error, canceled) are excluded.
 */
const RESUMABLE_STATUSES = new Set([
  "queued",
  "processing",
  "training",
  "scoring",
]);

// 1. Get Signed URL for Upload
router.post("/upload-url", requireAuth, uploadUrlLimiter, validateUploadUrl, async (req, res) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();

    const gcsFileName = generateSafeFilename((req as any).user.id);

    // V4 signed URLs include Content-Type in the canonical request.
    // Pass through the client's declared type so the PUT matches the signed URL.
    const signOptions: GetSignedUrlConfig = {
      version: "v4",
      action: "write",
      expires: Date.now() + 15 * 60 * 1000,
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
router.post("/start-job", requireAuth, uploadLimiter, validateStartJob, async (req, res) => {
  try {
    const { jobId, gcsFileName } = req.body;
    const userId = (req as any).user.id;

    // Authorization: ensure the caller owns the file path.
    const expectedPrefix = `uploads/${userId}/`;
    if (typeof gcsFileName !== 'string' || !gcsFileName.startsWith(expectedPrefix)) {
      logger.warn("start-job rejected: gcsFileName outside caller namespace", {
        jobId, userId, gcsFileName,
      });
      return res.status(403).json({ error: "Forbidden: file path does not belong to caller" });
    }

    const messageJson = {
      jobId: jobId,
      bucket: BUCKET_NAME,
      file: gcsFileName,
    };

    const dataBuffer = Buffer.from(JSON.stringify(messageJson));

    // Atomically claim the job ID. create() throws ALREADY_EXISTS if taken,
    // preventing overwrite of another user's job doc.
    // Status MUST be "queued" (matches worker JobStatus.QUEUED).
    let claimedExistingDoc = false;
    let existingStatus: string | undefined;
    try {
      await firestore.collection("jobs").doc(jobId).create({
        status: "queued",
        createdAt: new Date(),
        userId,
        gcsFileName,
      });
    } catch (createErr: any) {
      if (createErr?.code !== 6) {
        throw createErr;
      }

      // Same-user retry recovery: reuse doc if owned by caller, non-terminal,
      // and targeting the same file.
      const existingDoc = await firestore.collection("jobs").doc(jobId).get();
      const existing = existingDoc.exists ? existingDoc.data() : undefined;
      if (
        existing &&
        existing.userId === userId &&
        typeof existing.status === "string" &&
        RESUMABLE_STATUSES.has(existing.status) &&
        existing.gcsFileName === gcsFileName
      ) {
        claimedExistingDoc = true;
        existingStatus = existing.status;
        logger.info("start-job: reusing pre-existing non-terminal doc for retry", {
          jobId, userId, gcsFileName, existingStatus,
        });
      } else {
        logger.warn("start-job rejected: jobId already claimed", {
          jobId, userId,
          existingOwner: existing?.userId,
          existingStatus: existing?.status,
          existingGcsFileName: existing?.gcsFileName,
          requestedGcsFileName: gcsFileName,
        });
        return res.status(409).json({ error: "Job ID already exists" });
      }
    }

    // Publish only when useful. If the job is already past "queued", skip.
    const shouldPublish = !claimedExistingDoc || existingStatus === "queued";
    if (shouldPublish) {
      await pubsub.topic(TOPIC_ID).publishMessage({ data: dataBuffer });
    } else {
      logger.info("start-job: skipping republish for already-running job", {
        jobId, userId, existingStatus,
      });
    }

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
router.get("/job-status/:id", requireAuth, pollLimiter, validateJobId, async (req, res) => {
  try {
    const { id } = req.params;
    const userId = (req as any).user.id;
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ status: "not_found" });
    }

    const data = doc.data();

    // Only the job owner can read status/results.
    if (!data || data.userId !== userId) {
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

// 4. Export Outlier Results as CSV
router.get("/:id/export", requireAuth, downloadLimiter, validateJobId, async (req, res) => {
  try {
    const { id } = req.params;
    const doc = await firestore.collection("jobs").doc(id).get();
    if (!doc.exists) return res.status(404).json({ error: 'Job not found' });

    const job = doc.data();
    if (job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }
    if (job.status !== 'complete') {
      return res.status(400).json({ error: 'Job not complete' });
    }

    // Server-side expiry check.
    if (job.filesExpired) {
      return res.status(410).json({ error: 'Job files have expired and are no longer available for download' });
    }
    const createdAt = job.createdAt?.toDate ? job.createdAt.toDate() : new Date(job.createdAt);
    if (createdAt instanceof Date && !isNaN(createdAt.getTime())) {
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - 7);
      if (createdAt < cutoff) {
        return res.status(410).json({ error: 'Job files have expired and are no longer available for download' });
      }
    }

    const outliers = job.outliers || [];
    res.attachment(`outliers-${id}.csv`);

    // Outlier records separate user data (`data` sub-object) from system metadata.
    // Include reconstruction_error in export even if stored at top level only.
    const firstRow = outliers.length > 0 ? (outliers[0].data || outliers[0]) : {};
    const columnKeys: string[] = Object.keys(firstRow);
    if (!columnKeys.includes('reconstruction_error') && outliers.length > 0 && outliers[0].reconstruction_error != null) {
      columnKeys.push('reconstruction_error');
    }
    const sanitizedHeaders = columnKeys.map((key) => String(sanitizeFormulaInjection(key)));

    const csvStream = format({ headers: sanitizedHeaders });
    csvStream.pipe(res);
    outliers.forEach((row: any) => {
      const rowData = row.data || row;
      csvStream.write(columnKeys.map((key) => {
        const val = rowData[key] ?? row[key];
        return sanitizeFormulaInjection(val);
      }));
    });
    csvStream.end();
  } catch (error) {
    logger.error("Error exporting CSV", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.params.id,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to export CSV" });
  }
});

// 5. Cancel Job with Full Resource Cleanup (transactional)
router.delete("/:id", requireAuth, validateJobId, async (req, res) => {
  const { id } = req.params;
  const userId = (req as any).user.id;
  const TERMINAL_STATUSES = new Set(['complete', 'error', 'canceled']);
  let gcsFileName: string | undefined;
  let vertexJobName: string | undefined;

  try {
    await firestore.runTransaction(async (tx) => {
      const docRef = firestore.collection("jobs").doc(id);
      const snap = await tx.get(docRef);
      if (!snap.exists) throw new Error('NOT_FOUND');
      const job = snap.data() || {};
      if (job.userId !== userId) throw new Error('NOT_FOUND');
      if (TERMINAL_STATUSES.has(job.status)) throw new Error(`TERMINAL:${job.status}`);
      gcsFileName = job.gcsFileName || job.gcsPath || job.file;
      vertexJobName = job.vertexJobName;
      tx.update(docRef, { status: 'canceled', canceledAt: new Date() });
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    if (msg === 'NOT_FOUND') return res.status(404).json({ error: 'Job not found' });
    if (msg.startsWith('TERMINAL:')) {
      return res.status(409).json({
        error: `Cannot cancel job in terminal state: ${msg.slice('TERMINAL:'.length)}`,
      });
    }
    logger.error("Error canceling job (transaction)", { error: msg, jobId: id, userId });
    return res.status(500).json({ error: "Failed to cancel job" });
  }

  try {
    if (gcsFileName) {
      await storage.bucket(BUCKET_NAME).file(gcsFileName).delete();
      logger.info('Deleted GCS file for canceled job', { jobId: id, file: gcsFileName });
    }
  } catch (error) {
    logger.warn('Failed to delete GCS file (continuing cleanup)', {
      jobId: id, file: gcsFileName,
      error: error instanceof Error ? error.message : String(error),
    });
  }

  await cancelVertexAIJob(vertexJobName, id);
  logger.info('Job canceled successfully', { jobId: id, userId });
  res.json({ success: true, message: 'Job canceled and resources cleaned up' });
});

// 6. Manual File Deletion (completed/errored/canceled jobs)
router.delete("/:id/files", requireAuth, validateJobId, async (req, res) => {
  try {
    const { id } = req.params;
    const doc = await firestore.collection("jobs").doc(id).get();
    if (!doc.exists) return res.status(404).json({ error: 'Job not found' });

    const job = doc.data();
    if (job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const DELETABLE_STATUSES = new Set(['complete', 'error', 'canceled']);
    if (!DELETABLE_STATUSES.has(job.status)) {
      return res.status(400).json({ error: 'Cannot delete files from running job. Cancel job first.' });
    }

    const gcsFileName = job.gcsFileName || job.gcsPath || job.file;
    let filesDeleted = 0;
    let uploadDeleteFailed = false;

    if (gcsFileName) {
      try {
        await storage.bucket(BUCKET_NAME).file(gcsFileName).delete();
        filesDeleted++;
      } catch (error) {
        if (isGcsNotFoundError(error)) {
          logger.info('GCS upload file already absent', { jobId: id, file: gcsFileName });
        } else {
          uploadDeleteFailed = true;
          logger.warn('Failed to delete GCS upload file', {
            jobId: id, file: gcsFileName,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    const resultFileName = `results/${id}.json`;
    let resultDeleteFailed = false;
    try {
      await storage.bucket(BUCKET_NAME).file(resultFileName).delete();
      filesDeleted++;
    } catch (error) {
      if (isGcsNotFoundError(error)) {
        logger.info('GCS result file already absent', { jobId: id, file: resultFileName });
      } else {
        resultDeleteFailed = true;
        logger.warn('Failed to delete GCS result file', {
          jobId: id, file: resultFileName,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    if ((gcsFileName && uploadDeleteFailed) || resultDeleteFailed) {
      return res.status(502).json({
        error: 'Failed to delete one or more files from storage. Please retry.',
        filesDeleted,
      });
    }

    await firestore.collection("jobs").doc(id).update({
      filesExpired: true,
      filesDeletedAt: new Date()
    });

    res.json({ success: true, message: 'Files deleted successfully', filesDeleted });
  } catch (error) {
    logger.error("Error deleting job files", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.params.id,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to delete files" });
  }
});

export const jobsRouter = router;
