/**
 * Jobs API Routes
 */

import { Router, Request, Response } from "express";
import { v4 as uuidv4 } from "uuid";
import { format } from 'fast-csv';
import { requireAuth } from '../middleware/auth';
import { uploadLimiter, pollLimiter, downloadLimiter } from '../middleware/rateLimits';
import { validateJobId, validateUploadUrl, validateStartJob } from '../middleware/validation';
import { generateSafeFilename } from '../utils/fileValidation';
import { sanitizeFormulaInjection } from '../utils/csvSanitization';
import { logger } from '../config/logger';
import { storage, firestore, pubsub } from '../config/gcp-clients';
import { cancelVertexAIJob } from '../services/vertexAi';

function isGcsNotFoundError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const e = error as { code?: unknown };
  return e.code === 404 || e.code === 'ENOENT';
}

const router = Router();

const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "your-bucket-name";
const TOPIC_ID = process.env.PUBSUB_TOPIC_ID || "your-topic-id";

router.post("/upload-url", requireAuth, uploadLimiter, validateUploadUrl, async (req: Request, res: Response) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();

    if (contentType && contentType !== 'text/csv' && contentType !== 'application/vnd.ms-excel') {
      logger.warn('Unexpected content type for CSV upload', {
        contentType,
        filename,
        userId: (req as any).user.id
      });
    }

    const gcsFileName = generateSafeFilename((req as any).user.id);

    const [url] = await storage
      .bucket(BUCKET_NAME)
      .file(gcsFileName)
      .getSignedUrl({
        version: "v4",
        action: "write",
        expires: Date.now() + 15 * 60 * 1000,
        contentType: 'text/csv',
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

router.post("/start-job", requireAuth, uploadLimiter, validateStartJob, async (req: Request, res: Response) => {
  try {
    const { jobId, gcsFileName } = req.body;
    const userId = (req as any).user.id;

    // Validate that gcsFileName lives within this user's upload namespace.
    const expectedPrefix = `uploads/${userId}/`;
    if (
      !gcsFileName ||
      typeof gcsFileName !== 'string' ||
      !gcsFileName.startsWith(expectedPrefix) ||
      !gcsFileName.endsWith('.csv')
    ) {
      return res.status(400).json({ error: 'Invalid file reference' });
    }

    const messageJson = {
      jobId: jobId,
      bucket: BUCKET_NAME,
      file: gcsFileName,
    };

    const dataBuffer = Buffer.from(JSON.stringify(messageJson));
    await pubsub.topic(TOPIC_ID).publishMessage({ data: dataBuffer });

    await firestore.collection("jobs").doc(jobId).set({
      status: "uploading",
      createdAt: new Date(),
      userId,
      gcsFileName,
      bucket: BUCKET_NAME,
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

router.get("/job-status/:id", requireAuth, pollLimiter, validateJobId, async (req: Request, res: Response) => {
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

router.get("/:id/export", requireAuth, downloadLimiter, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const job = doc.data();

    if (job.userId && job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }

    if (job.status !== 'complete') {
      return res.status(400).json({ error: 'Job not complete' });
    }

    // Server-side expiry check. The UI hides the download button after
    // 7 days / manual deletion, but a direct API call would still serve
    // the data. Block that here with a 410 Gone.
    if (job.filesExpired) {
      return res.status(410).json({ error: 'Job files have expired and are no longer available for download' });
    }
    // Also enforce the 7-day retention window. Firestore Admin SDK
    // returns createdAt as a Timestamp with .toDate().
    const createdAt = job.createdAt?.toDate
      ? job.createdAt.toDate()
      : new Date(job.createdAt);
    if (createdAt instanceof Date && !isNaN(createdAt.getTime())) {
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - 7);
      if (createdAt < cutoff) {
        return res.status(410).json({ error: 'Job files have expired and are no longer available for download' });
      }
    }

    const outliers = job.outliers || [];

    res.attachment(`outliers-${id}.csv`);

    const RESERVED_OUTLIER_KEYS = new Set(['__meta']);

    const columnKeys: string[] =
      outliers.length > 0
        ? Object.keys(outliers[0]).filter((k) => !RESERVED_OUTLIER_KEYS.has(k))
        : [];

    const sanitizedHeaders = columnKeys.map(
      (key) => String(sanitizeFormulaInjection(key))
    );

    const csvStream = format({ headers: sanitizedHeaders });
    csvStream.pipe(res);

    outliers.forEach((row: any) => {
      const values = columnKeys.map((key) =>
        sanitizeFormulaInjection(row[key])
      );
      csvStream.write(values);
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

router.delete("/:id", requireAuth, validateJobId, async (req: Request, res: Response) => {
  const { id } = req.params;
  const userId = (req as any).user.id;

  const TERMINAL_STATUSES = new Set(['complete', 'error', 'canceled']);
  let gcsFileName: string | undefined;
  let vertexJobName: string | undefined;

  try {
    await firestore.runTransaction(async (tx) => {
      const docRef = firestore.collection("jobs").doc(id);
      const snap = await tx.get(docRef);

      if (!snap.exists) {
        throw new Error('NOT_FOUND');
      }
      const job = snap.data() || {};

      if (job.userId && job.userId !== userId) {
        throw new Error('NOT_FOUND');
      }
      if (TERMINAL_STATUSES.has(job.status)) {
        throw new Error(`TERMINAL:${job.status}`);
      }

      gcsFileName = job.gcsFileName || job.file;
      vertexJobName = job.vertexJobName;

      tx.update(docRef, {
        status: 'canceled',
        canceledAt: new Date(),
      });
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    if (msg === 'NOT_FOUND') {
      return res.status(404).json({ error: 'Job not found' });
    }
    if (msg.startsWith('TERMINAL:')) {
      return res.status(409).json({
        error: `Cannot cancel job in terminal state: ${msg.slice('TERMINAL:'.length)}`,
      });
    }
    logger.error("Error canceling job (transaction)", {
      error: msg,
      jobId: id,
      userId,
    });
    return res.status(500).json({ error: "Failed to cancel job" });
  }

  try {
    if (gcsFileName) {
      await storage.bucket(BUCKET_NAME).file(gcsFileName).delete();
      logger.info('Deleted GCS file for canceled job', { jobId: id, file: gcsFileName });
    }
  } catch (error) {
    logger.warn('Failed to delete GCS file (continuing cleanup)', {
      jobId: id,
      file: gcsFileName,
      error: error instanceof Error ? error.message : String(error),
    });
  }

  await cancelVertexAIJob(vertexJobName, id);

  logger.info('Job canceled successfully', { jobId: id, userId });
  res.json({ success: true, message: 'Job canceled and resources cleaned up' });
});

router.delete("/:id/files", requireAuth, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const job = doc.data();

    if (job.userId && job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const DELETABLE_STATUSES = new Set(['complete', 'error', 'canceled']);
    if (!DELETABLE_STATUSES.has(job.status)) {
      return res.status(400).json({ error: 'Cannot delete files from running job. Cancel job first.' });
    }

    const gcsFileName = job.gcsFileName || job.file;
    let filesDeleted = 0;
    let uploadDeleteFailed = false;
    let resultDeleteFailed = false;

    if (gcsFileName) {
      try {
        await storage.bucket(BUCKET_NAME).file(gcsFileName).delete();
        filesDeleted++;
        logger.info('Deleted GCS upload file', { jobId: id, file: gcsFileName });
      } catch (error) {
        if (isGcsNotFoundError(error)) {
          logger.info('GCS upload file already absent, treating as deleted', {
            jobId: id,
            file: gcsFileName,
          });
        } else {
          uploadDeleteFailed = true;
          logger.warn('Failed to delete GCS upload file', {
            jobId: id,
            file: gcsFileName,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    const resultFileName = `results/${id}.json`;
    try {
      await storage.bucket(BUCKET_NAME).file(resultFileName).delete();
      filesDeleted++;
      logger.info('Deleted GCS result file', { jobId: id, file: resultFileName });
    } catch (error) {
      if (isGcsNotFoundError(error)) {
        logger.info('GCS result file already absent, treating as deleted', {
          jobId: id,
          file: resultFileName,
        });
      } else {
        resultDeleteFailed = true;
        logger.warn('Failed to delete GCS result file', {
          jobId: id,
          file: resultFileName,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    if (gcsFileName && uploadDeleteFailed) {
      return res.status(502).json({
        error: 'Failed to delete uploaded file from storage. Please retry.',
        filesDeleted,
      });
    }

    await firestore.collection("jobs").doc(id).update({
      filesExpired: true,
      filesDeletedAt: new Date()
    });

    logger.info('Job files deleted manually', {
      jobId: id,
      filesDeleted,
      resultDeleteFailed,
      userId: (req as any).user?.id,
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
