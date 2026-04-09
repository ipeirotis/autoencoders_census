/**
 * Jobs API Routes - Handles the 3-step upload workflow:
 *
 * 1. POST /upload-url  - Generates a signed GCS URL for direct browser upload
 * 2. POST /start-job   - Creates Firestore doc + publishes Pub/Sub message to trigger worker
 * 3. GET /job-status/:id - Polls Firestore for job status and results
 *
 * Flow: Frontend → (signed URL) → GCS → Pub/Sub → worker.py → Vertex AI → Firestore → Frontend polls
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

/**
 * Returns true when a Google Cloud Storage delete error represents
 * "object already gone" - either a 404 from the GCS API or the ENOENT code
 * some clients surface. Treating already-deleted as success keeps the manual
 * cleanup endpoint idempotent (e.g. the lifecycle rule beat the user to it).
 */
function isGcsNotFoundError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const e = error as { code?: unknown };
  return e.code === 404 || e.code === 'ENOENT';
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

    // Initialize Firestore document.
    // gcsFileName is persisted so the cancel and delete-files endpoints can
    // find the upload object in GCS later - without this the cleanup paths
    // silently skip the GCS delete and leave the upload behind.
    await firestore.collection("jobs").doc(jobId).set({
      status: "uploading", // Initial state
      createdAt: new Date(),
      userId: (req as any).user.id,
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

// 3. Check Job Status
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

// 4. Export Outlier Results as CSV
// Mounted at /api/jobs, so this becomes GET /api/jobs/:id/export
router.get("/:id/export", requireAuth, downloadLimiter, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    // Fetch job from Firestore
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const job = doc.data();

    // Enforce job ownership - prevent users from exporting other users' data
    if (job.userId && job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Only export completed jobs
    if (job.status !== 'complete') {
      return res.status(400).json({ error: 'Job not complete' });
    }

    const outliers = job.outliers || [];

    // Set CSV download headers
    res.attachment(`outliers-${id}.csv`);

    // Stream CSV with sanitization.
    //
    // IMPORTANT: Sanitize *both* the keys and the values. The keys originate
    // from user-uploaded column names, and with `headers: true` fast-csv
    // picks them up as CSV header cells verbatim. A header like "=SUM(..)"
    // would otherwise evaluate as a formula when the file is opened in a
    // spreadsheet client.
    const csvStream = format({ headers: true });
    csvStream.pipe(res);

    outliers.forEach((row: any) => {
      const sanitizedRow = Object.fromEntries(
        Object.entries(row).map(([key, value]) => [
          sanitizeFormulaInjection(key),
          sanitizeFormulaInjection(value)
        ])
      );
      csvStream.write(sanitizedRow);
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

// 5. Cancel Job with Full Resource Cleanup
// Mounted at /api/jobs, so this becomes DELETE /api/jobs/:id
router.delete("/:id", requireAuth, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    // Fetch job to get GCS file path
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const job = doc.data();

    // Enforce job ownership - prevent users from canceling other users' jobs
    if (job.userId && job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Refuse to "cancel" a job that has already reached a terminal state.
    // Without this guard a stale client tab (or direct API call) could flip
    // an already-`complete` job to `canceled`, delete its storage artifacts,
    // and silently invalidate finished results.
    const TERMINAL_STATUSES = new Set(['complete', 'error', 'canceled']);
    if (TERMINAL_STATUSES.has(job.status)) {
      return res.status(409).json({
        error: `Cannot cancel job in terminal state: ${job.status}`,
      });
    }

    const gcsFileName = job.gcsFileName || job.file; // Check both possible fields

    // 1. Delete GCS uploaded file (best-effort)
    try {
      if (gcsFileName) {
        await storage.bucket(BUCKET_NAME).file(gcsFileName).delete();
        logger.info('Deleted GCS file for canceled job', { jobId: id, file: gcsFileName });
      }
    } catch (error) {
      logger.warn('Failed to delete GCS file (continuing cleanup)', {
        jobId: id,
        file: gcsFileName,
        error: error instanceof Error ? error.message : String(error)
      });
      // Continue with other cleanup steps
    }

    // 2. Cancel Vertex AI job (best-effort, async)
    // Note: May not stop job if already running, cancellation is best-effort.
    // Use the actual server-generated Vertex resource name stored when the job
    // was dispatched. If absent (e.g. local mode or job never reached Vertex),
    // cancelVertexAIJob will skip the call.
    await cancelVertexAIJob(job.vertexJobName, id);

    // 3. Update Firestore status to "canceled"
    await firestore.collection("jobs").doc(id).update({
      status: 'canceled',
      canceledAt: new Date()
    });

    logger.info('Job canceled successfully', { jobId: id, userId: (req as any).user?.id });

    res.json({ success: true, message: 'Job canceled and resources cleaned up' });
  } catch (error) {
    logger.error("Error canceling job", {
      error: error instanceof Error ? error.message : String(error),
      jobId: req.params.id,
      userId: (req as any).user?.id
    });
    res.status(500).json({ error: "Failed to cancel job" });
  }
});

// 6. Manual File Deletion (Completed Jobs)
// Separate from cancellation endpoint (which is for running jobs)
// Mounted at /api/jobs, so this becomes DELETE /api/jobs/:id/files
router.delete("/:id/files", requireAuth, validateJobId, async (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    // Fetch job
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const job = doc.data();

    // Enforce job ownership - prevent users from deleting other users' files
    if (job.userId && job.userId !== (req as any).user.id) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // Only allow deletion of terminal jobs (not running jobs).
    // NOTE: worker.py writes failures as JobStatus.ERROR ("error"), not
    // "failed" - including "failed" in the list previously would have
    // rejected real failed jobs and left their artifacts orphaned.
    const DELETABLE_STATUSES = new Set(['complete', 'error', 'canceled']);
    if (!DELETABLE_STATUSES.has(job.status)) {
      return res.status(400).json({ error: 'Cannot delete files from running job. Cancel job first.' });
    }

    const gcsFileName = job.gcsFileName || job.file;
    let filesDeleted = 0;
    let uploadDeleteFailed = false;
    let resultDeleteFailed = false;

    // Delete GCS uploaded file (best-effort).
    // A 404 from GCS is not a failure: the object is already gone (lifecycle
    // rule, previous cleanup, etc.), so cleanup is effectively complete.
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

    // Delete result files if exist (best-effort).
    // Same 404-is-success semantics - the results file may simply never have
    // been written (e.g. error'd jobs).
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

    // Only mark the job as expired if the upload object was either deleted
    // now or already absent. If the upload delete threw a real error (not
    // 404), surface a 502 so the user can retry - we can't flip
    // filesExpired to true while the CSV is still sitting in GCS.
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
