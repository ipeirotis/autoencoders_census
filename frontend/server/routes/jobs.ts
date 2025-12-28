import { Router } from "express";
import { Storage } from "@google-cloud/storage";
import { Firestore } from "@google-cloud/firestore";
import { PubSub } from "@google-cloud/pubsub";
import { v4 as uuidv4 } from "uuid";

const router = Router();
const storage = new Storage();
const firestore = new Firestore();
const pubsub = new PubSub();

// Configuration from Environment Variables
const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "your-bucket-name";
const TOPIC_ID = process.env.PUBSUB_TOPIC_ID || "your-topic-id";

// 1. Get Signed URL for Upload
router.post("/upload-url", async (req, res) => {
  try {
    const { filename, contentType } = req.body;
    const jobId = uuidv4();
    const gcsFileName = `uploads/${jobId}/${filename}`;

    const [url] = await storage
      .bucket(BUCKET_NAME)
      .file(gcsFileName)
      .getSignedUrl({
        version: "v4",
        action: "write",
        expires: Date.now() + 15 * 60 * 1000, // 15 minutes
        contentType,
      });

    res.json({ url, jobId, gcsFileName });
  } catch (error) {
    console.error("Error generating signed URL:", error);
    res.status(500).json({ error: "Failed to generate upload URL" });
  }
});

// 2. Start Processing (Trigger Pub/Sub)
router.post("/start-job", async (req, res) => {
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
    await firestore.collection("jobs").doc(jobId).set({
      status: "uploading", // Initial state
      createdAt: new Date(),
    });

    res.json({ success: true, message: "Job started" });
  } catch (error) {
    console.error("Error starting job:", error);
    res.status(500).json({ error: "Failed to start job" });
  }
});

// 3. Check Job Status
router.get("/job-status/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const doc = await firestore.collection("jobs").doc(id).get();

    if (!doc.exists) {
      return res.status(404).json({ status: "not_found" });
    }

    const data = doc.data();
    res.json(data);
  } catch (error) {
    console.error("Error checking status:", error);
    res.status(500).json({ error: "Failed to check status" });
  }
});

export const jobsRouter = router;