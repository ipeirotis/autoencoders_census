/**
 * Server Entry Point - Starts the Express API server.
 *
 * Port is configured via PORT environment variable (defaults to 5001).
 * Environment validation happens at import time via index.ts.
 *
 * Usage: npx tsx server/start.ts
 */

import { createServer } from "./index";
import { env } from "./config/env";
import { logger } from "./config/logger";

const app = createServer();

app.listen(env.PORT, () => {
  logger.info(`Backend API running on http://localhost:${env.PORT}`, {
    nodeEnv: env.NODE_ENV,
    project: env.GOOGLE_CLOUD_PROJECT,
  });
});