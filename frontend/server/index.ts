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
      const uniqueId = `job_${Date.now()}_${Math.random().toString(36).substring(7)}`;

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
      const jobMetadata = {
        jobId: uniqueId,
        fileName: originalName,
        gcsPath: safeFilename,
        bucket: GCS_BUCKET_NAME,
        status: "uploaded", // Initial status
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

  // --- NEW: Job Status Route ---
  app.get("/api/jobs/job-status/:jobId", async (req, res) => {
    try {
      const { jobId } = req.params;
      const doc = await firestore.collection("jobs").doc(jobId).get();

      if (!doc.exists) {
        // If the doc doesn't exist yet, it might be just starting
        return res.json({ status: "uploading" });
      }

      const data = doc.data();
      res.json(data); // Returns { status: 'complete', outliers: [...] }
    } catch (error) {
      logger.error("Error checking status", { error: error instanceof Error ? error.message : String(error) });
      res.status(500).json({ error: "Failed to check status" });
    }
  });

  // --- Error Handler Middleware (MUST be last) ---
  // Catches all unhandled errors from routes above
  // Logs full details server-side, returns generic message in production
  app.use(errorHandler);

  return app;
}