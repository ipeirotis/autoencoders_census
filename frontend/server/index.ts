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
import { storage, firestore, pubsub } from "./config/gcp-clients";
import path from "path";

// --- Configuration ---
const GCS_BUCKET_NAME = env.GCS_BUCKET_NAME;
const PUBSUB_TOPIC_NAME = "job-upload-topic";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

export function createServer() {
  const app = express();

  app.use(corsConfig);
  app.use(helmetConfig);

  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  app.use(sessionConfig);
  app.use(passport.initialize());
  app.use(passport.session());

  app.get("/api/ping", (_req, res) => res.json({ message: "pong" }));

  app.use("/api/auth", authRouter);
  app.use("/api/jobs", jobsRouter);

  app.post("/api/upload", requireAuth, uploadLimiter, upload.single("file"), async (req, res, next) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const validation = await validateCSVContent(req.file.buffer);
      if (!validation.valid) {
        logger.warn('CSV validation failed', {
          reason: validation.reason,
          userId: (req as any).user?.id
        });
        return res.status(400).json({ error: 'Invalid CSV file format' });
      }

      const originalName = req.file.originalname;
      const uniqueId = `job_${Date.now()}_${Math.random().toString(36).substring(7)}`;
      const safeFilename = generateSafeFilename((req as any).user.id);

      logger.info(`Starting upload job: ${uniqueId}`, {
        userId: (req as any).user.id,
        originalFilename: originalName
      });

      const bucket = storage.bucket(GCS_BUCKET_NAME);
      const file = bucket.file(safeFilename);

      await file.save(req.file.buffer, {
        contentType: req.file.mimetype,
        resumable: false,
      });

      logger.info(`File saved to GCS: ${safeFilename}`);

      const jobMetadata = {
        jobId: uniqueId,
        fileName: originalName,
        gcsPath: safeFilename,
        bucket: GCS_BUCKET_NAME,
        status: "uploaded",
        createdAt: new Date().toISOString(),
        userId: (req as any).user.id,
      };

      await firestore.collection("jobs").doc(uniqueId).set(jobMetadata);
      logger.info(`Firestore document created: jobs/${uniqueId}`);

      const messageBuffer = Buffer.from(JSON.stringify({
        jobId: uniqueId,
        bucket: GCS_BUCKET_NAME,
        file: safeFilename
      }));

      try {
        await pubsub.topic(PUBSUB_TOPIC_NAME).publishMessage({ data: messageBuffer });
        logger.info(`Pub/Sub message sent to topic: ${PUBSUB_TOPIC_NAME}`);
      } catch (pubError) {
        logger.warn("Failed to publish to Pub/Sub", { error: pubError });
      }

      res.json({
        dataset_id: uniqueId,
        message: "Upload successful. Processing started.",
      });

    } catch (error) {
      next(error);
    }
  });

  app.use(errorHandler);

  return app;
}
