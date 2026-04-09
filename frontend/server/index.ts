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
import { corsConfig, helmetConfig, csrfOriginCheck } from "./middleware/security";
import { errorHandler } from "./middleware/errorHandler";
import { requireAuth } from "./middleware/auth";
import { uploadLimiter } from "./middleware/rateLimits";
import { validateCSVContent, generateSafeFilename } from "./utils/fileValidation";
import { sessionConfig } from "./config/session";
import { passport } from "./middleware/auth";
import { env } from "./config/env";
import { logger } from "./config/logger";
import { storage, firestore, pubsub } from "./config/gcp-clients";
import path from "path";

// --- Configuration ---
const GCS_BUCKET_NAME = env.GCS_BUCKET_NAME;
const PUBSUB_TOPIC_NAME = "job-upload-topic";

// --- Multer Middleware (Memory Storage) ---
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

export function createServer() {
  const app = express();

  if (env.NODE_ENV === "production") {
    app.set("trust proxy", 1);
  }

  // Security middleware - apply BEFORE routes
  app.use(corsConfig);
  app.use(csrfOriginCheck);
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
      }

      const originalName = req.file.originalname;
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
        resumable: false,
      });

      logger.info(`File saved to GCS: ${safeFilename}`);

      // 2. Create Firestore Document (Job Metadata)
      const jobMetadata = {
        jobId: uniqueId,
        fileName: originalName,
        gcsPath: safeFilename,
        bucket: GCS_BUCKET_NAME,
        status: "queued",
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
      }

      // 4. Return Response matching UploadResponse interface
      res.json({
        dataset_id: uniqueId,
        message: "Upload successful. Processing started.",
      });

    } catch (error) {
      next(error);
    }
  });

  // --- Error Handler Middleware (MUST be last) ---
  app.use(errorHandler);

  return app;
}
