/**
 * server/index.ts
 *
 * Express API server for the AutoEncoder app. This is where the GCS upload happens.
 *
 * Key responsibilities:
 * - Initialize Google Cloud clients (Storage, Firestore, PubSub)
 * - POST /api/upload endpoint that:
 *   1. Receives file via multer
 *   2. Uploads to GCS bucket
 *   3. Creates Firestore job document
 *   4. Publishes Pub/Sub message to trigger worker
 */

import "dotenv/config";
import express from "express";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import { Storage } from "@google-cloud/storage";
import { Firestore } from "@google-cloud/firestore";
import { PubSub } from "@google-cloud/pubsub";
import { jobsRouter } from "./routes/jobs";
import { authRouter } from "./routes/auth";
import { corsConfig, helmetConfig } from "./middleware/security";
import { errorHandler } from "./middleware/errorHandler";
import { requireAuth } from "./middleware/auth";
import { uploadLimiter } from "./middleware/rateLimits";
import { validateCSVContent, generateSafeFilename } from "./utils/fileValidation";
import { sessionConfig } from "./config/session";
import { passport } from "./middleware/auth";
import { env } from "./config/env";
import { logger } from "./config/logger";
import path from "path";

// --- Configuration ---
// Environment variables are validated at startup via env module import above
// Server will fail fast with clear error message if required vars are missing
const GCS_BUCKET_NAME = env.GCS_BUCKET_NAME;
const PROJECT_ID = env.GOOGLE_CLOUD_PROJECT;
const PUBSUB_TOPIC_NAME = "job-upload-topic"; // The topic your Worker listens to

// --- Google Cloud Clients ---
const storage = new Storage({ projectId: PROJECT_ID });
const firestore = new Firestore({ projectId: PROJECT_ID });
const pubsub = new PubSub({ projectId: PROJECT_ID });

// --- Multer Middleware (Memory Storage) ---
// Keeps the file in RAM (req.file.buffer) so we can stream it to GCS
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // Limit to 50MB
});

export function createServer() {
  const app = express();

  // Trust the first proxy hop in production so Express sees the client
  // protocol/IP forwarded by a TLS-terminating proxy (Cloud Run, Nginx, etc.).
  // Without this, `cookie.secure: true` on the session causes express-session
  // to refuse to issue session cookies (it sees req.secure === false), and
  // every subsequent authenticated request returns 401.
  if (env.NODE_ENV === "production") {
    app.set("trust proxy", 1);
  }

  // Security middleware - apply BEFORE routes
  app.use(corsConfig);
  app.use(helmetConfig);

  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Session and authentication middleware
  app.use(sessionConfig);
  app.use(passport.initialize());
  app.use(passport.session());

  // Health Check
  app.get("/api/ping", (_req, res) => res.json({ message: "pong" }));

  // Routes
  app.use("/api/auth", authRouter);
  app.use("/api/jobs", jobsRouter);

  // --- The Main Upload Route ---
  // (keep this as a fallback, but new frontend uses the 'jobsRouter' endpoints instead)
  app.post("/api/upload", requireAuth, uploadLimiter, upload.single("file"), async (req, res, next) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      // Validate CSV content
      const validation = await validateCSVContent(req.file.buffer);
      if (!validation.valid) {
        logger.warn('CSV validation failed', {
          reason: validation.reason,
          userId: (req as any).user?.id
        });
        return res.status(400).json({ error: 'Invalid CSV file format' });
        // Don't expose validation.reason to client (logged server-side)
      }

      const originalName = req.file.originalname;
      // Use a UUID v4 for the job id so it matches the `validateJobId` UUID
      // check on /api/jobs/job-status/:id. The previous
      // `job_<timestamp>_<random>` format got rejected at polling time,
      // leaving the fallback upload flow internally inconsistent.
      const uniqueId = uuidv4();

      // Generate safe filename (discard user-provided name)
      const safeFilename = generateSafeFilename((req as any).user.id);

      logger.info(`Starting upload job: ${uniqueId}`, {
        userId: (req as any).user.id,
        originalFilename: originalName
      });

      // 1. Stream file to Google Cloud Storage
      const bucket = storage.bucket(GCS_BUCKET_NAME);
      const file = bucket.file(safeFilename);

      await file.save(req.file.buffer, {
        contentType: req.file.mimetype,
        resumable: false, // Simple upload for small/medium files
      });

      logger.info(`File saved to GCS: ${safeFilename}`);

      // 2. Create Firestore Document (Job Metadata)
      //
      // Codex P1 (r3053812511): initial status MUST be "queued"
      // (JobStatus.QUEUED in worker.py). Any other first status
      // ("uploaded", "uploading", ...) would fail the worker's
      // is_valid_transition(None, ...) check, and the job would stay
      // stuck because update_job_status(..., PROCESSING) would raise
      // ValueError and the callback would ack the message without
      // running. See the identical fix in
      // frontend/server/routes/jobs.ts /start-job for context.
      const jobMetadata = {
        jobId: uniqueId,
        fileName: originalName,
        gcsPath: safeFilename,
        bucket: GCS_BUCKET_NAME,
        status: "queued", // Initial status (matches worker JobStatus.QUEUED)
        createdAt: new Date().toISOString(),
        userId: (req as any).user.id,
      };

      await firestore.collection("jobs").doc(uniqueId).set(jobMetadata);
      logger.info(`Firestore document created: jobs/${uniqueId}`);

      // 3. Publish Pub/Sub Message (Trigger the Worker)
      const messageBuffer = Buffer.from(JSON.stringify({
        jobId: uniqueId,
        bucket: GCS_BUCKET_NAME,
        file: safeFilename
      }));

      try {
        await pubsub.topic(PUBSUB_TOPIC_NAME).publishMessage({ data: messageBuffer });
        logger.info(`Pub/Sub message sent to topic: ${PUBSUB_TOPIC_NAME}`);
      } catch (pubError) {
        logger.warn("Failed to publish to Pub/Sub (is the local emulator running or topic missing?)", { error: pubError });
        // We don't fail the request here, just warn, so you can test upload without PubSub initially
      }

      // 4. Return Response matching UploadResponse interface
      res.json({
        dataset_id: uniqueId, // <--- CHANGED from jobId to dataset_id
        message: "Upload successful. Processing started.",
      });

    } catch (error) {
      next(error); // Pass to error handler
    }
  });

  // --- Error Handler Middleware (MUST be last) ---
  // Catches all unhandled errors from routes above
  // Logs full details server-side, returns generic message in production
  app.use(errorHandler);

  return app;
}
