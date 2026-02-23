import { Router } from "express";
import { Storage } from "@google-cloud/storage";
import { Firestore } from "@google-cloud/firestore";
import { v4 as uuidv4 } from "uuid";

const router = Router();
const storage = new Storage();
const firestore = new Firestore();
const BUCKET_NAME = process.env.GCS_BUCKET_NAME || "your-bucket-name";

// Endpoint 1: Generate Signed URL
router.post("/upload-url", async (req, res) => {
  const { filename, contentType } = req.body;
  const jobId = uuidv4(); // Create a unique ID for this job
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

  res.json({ url, id: jobId });
});

// Endpoint 2: Check Job Status (Reads Firestore)
router.get("/job-status/:id", async (req, res) => {
  const { id } = req.params;
  const doc = await firestore.collection("jobs").doc(id).get();

  if (!doc.exists) {
    return res.json({ status: "processing" }); // Worker hasn't written yet
  }

  const data = doc.data();
  // Assuming your worker writes { status: 'done', results: [...] }
  res.json(data);
});

export default router;