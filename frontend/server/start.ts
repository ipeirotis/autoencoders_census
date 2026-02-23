/**
 * Server Entry Point - Starts the Express API server on port 3000.
 *
 * Usage: npx tsx server/start.ts
 */

import { createServer } from "./index";

const port = 3000;
const app = createServer();

app.listen(port, () => {
  console.log(`✅ Backend API running on http://localhost:${port}`);
});