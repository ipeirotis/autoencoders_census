import "dotenv/config";
import express from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import { _readonly } from "zod/v4/core";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Health check
  // Example API routes
  app.get("/api/ping", (_req, res) => {
    console.log("✅ Ping endpoint hit");
    res.json({ message: "pong" });
  });

  app.get("/api/demo", (_req, res) => {
    res.json({ message: "This is a demo endpoint." });
  });

  // Upload endpoint
  app.post("/api/upload", async (req, res) => {
    console.log("Upload request received");
    console.log("Headers:", req.headers);
    console.log("Body:", req.body);
    try {
      // mock data for now
      // TODO: Connect to autoencoder backend

      const mockResponse = {
        dataset_id: `dataset_${Date.now()}`,
        schema: [
          { name: "column1", detected_type: "string" },
          { name: "column2", detected_type: "number" }
        ],
        preview: [
          { column1: "value1", column2: 123 },
          { column1: "value2", column2: 456 }
        ]
      };
      res.json(mockResponse);
    } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({ 
        error: error instanceof Error ? error.message : "Failed to process upload." 
      });
    }
  });

  // ✅ SAFE way to log routes (optional - you can remove this entirely)
  console.log("📋 Registered routes:");
  console.log("  GET /api/ping");
  console.log("  POST /api/upload");

  return app;
}
