# Technology Stack - Production Hardening

**Project:** AutoEncoder Outlier Detection Platform (v1.0 Production-Ready)
**Researched:** 2026-03-24

## Executive Summary

This document focuses exclusively on **production-hardening additions** to the existing validated stack (React 18, Vite, Express.js, Python/TensorFlow, Google Cloud). The recommended additions prioritize battle-tested, minimal-configuration libraries that integrate seamlessly with the current architecture.

**Philosophy:** Use focused libraries that solve one problem well. Avoid frameworks that require architectural changes.

---

## Recommended Stack Additions

### Security & Validation (Backend)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **express-rate-limit** | 8.3.1 | API rate limiting | De facto standard for Express rate limiting. 15M+ weekly downloads, actively maintained (last update: 15 days ago). Supports flexible algorithms (fixed window, sliding window, token bucket). Zero dependencies beyond Express. |
| **express-validator** | 7.3.1 | Input validation & sanitization | Built on validator.js. Chainable API integrates naturally with Express routes. Validates request bodies, query params, headers, and cookies. Essential for preventing XSS, SQL injection, and malformed input. |
| **helmet** | Latest (5.x+) | Security headers | One-line configuration sets 13-15 HTTP security headers (Content-Security-Policy, Strict-Transport-Security, X-Frame-Options, etc.). Official Express security recommendation. |
| **cors** | 2.8.5 | CORS configuration | Already installed. **Action required:** Replace `app.use(cors())` with origin allowlist configuration for production. |
| **csv-parse** | Latest (5.x+) | CSV streaming | Part of node-csv package. Stream-based Transform API prevents memory crashes on large files. Handles encoding issues, unicode, malformed CSV. Preferred over PapaParse for backend (server-side). |

### Security & Validation (Frontend)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **react-error-boundary** | 6.1.1 | Error boundaries | Reusable error boundary component with reset/retry support. Eliminates need to write class components. Supports functional components, hooks, and React 18+. Clean declarative approach recommended in 2026. |
| **papaparse** or **react-papaparse** | Latest (5.x+) | CSV preview/validation | Frontend CSV parsing for file preview before upload. Supports streaming, worker threads, and error handling. 5M+ weekly downloads. Alternative: react-papaparse wrapper for React-specific hooks. |

### TypeScript Tooling

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **TypeScript strict mode** | (config change) | Type safety | Already using TypeScript 5.9.3. Enable strict mode incrementally: start with `noImplicitAny`, then `strictNullChecks`, then `strict: true`. Prevents production bugs caught at compile time. |
| **typescript-strict-plugin** | Latest (optional) | Gradual strict migration | Allows strict mode per-directory using `// @ts-strict-ignore` comments. Enables strict for new code without fixing entire codebase. Recommended for incremental migration. |

### Development & Operational

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **dotenv** | 17.2.3 | Environment variables | Already installed. **Action required:** Add validation at startup (check for required vars like `GCS_BUCKET_NAME`, `GOOGLE_CLOUD_PROJECT`). |

---

## Already Installed (No Action Needed)

These packages are already in `frontend/package.json` and support production features:

| Package | Current Version | Used For |
|---------|----------------|----------|
| `@google-cloud/firestore` | 8.0.0 | Job progress tracking, metadata storage |
| `@google-cloud/pubsub` | 5.2.0 | Worker job queue |
| `@google-cloud/storage` | 7.18.0 | File uploads |
| `express` | 5.2.1 | API server (Note: Express 5 has built-in async error handling) |
| `multer` | 2.0.2 | File upload handling |
| `cors` | 2.8.5 | CORS (needs reconfiguration) |

**Note:** firebase-admin is NOT currently installed. The project uses individual `@google-cloud/*` packages, which is the correct approach for non-Firebase GCP projects.

---

## Installation Commands

### Backend (Express Server)

```bash
cd frontend  # Server code is in frontend/server/

# Security & validation
npm install express-rate-limit express-validator helmet

# CSV streaming (backend)
npm install csv-parse

# TypeScript types
npm install -D @types/express-rate-limit
```

### Frontend (React App)

```bash
cd frontend

# Error handling
npm install react-error-boundary

# CSV preview/validation (choose one)
npm install papaparse
# OR
npm install react-papaparse

# TypeScript types
npm install -D @types/papaparse
# OR (if using react-papaparse, types are included)
```

### Optional: Gradual Strict Mode

```bash
npm install -D typescript-strict-plugin
```

---

## Integration Points with Existing Stack

### Express Middleware Order (Critical)

Middleware must be applied in this specific order to work correctly:

```typescript
import express from "express";
import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";
import { body, validationResult } from "express-validator";

const app = express();

// 1. Security headers (FIRST - before any route handling)
app.use(helmet());

// 2. CORS (SECOND - before routes)
app.use(cors({
  origin: process.env.FRONTEND_URL || "http://localhost:5173",
  credentials: true
}));

// 3. Body parsing (BEFORE validation)
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 4. Rate limiting (BEFORE routes)
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
});
app.use("/api/", limiter);

// 5. Routes with validation
app.post("/api/upload",
  upload.single("file"),
  [
    body("file").custom((value, { req }) => {
      if (!req.file) throw new Error("No file uploaded");
      if (!req.file.originalname.endsWith(".csv")) {
        throw new Error("Only CSV files are allowed");
      }
      return true;
    }),
  ],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    // ... existing upload logic
  }
);

// 6. Error handler (LAST)
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Internal server error" });
});
```

### React Error Boundary Placement

```tsx
import { ErrorBoundary } from "react-error-boundary";

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onReset={() => window.location.reload()}
      onError={(error, errorInfo) => {
        console.error("Error caught by boundary:", error, errorInfo);
        // TODO: Send to monitoring service
      }}
    >
      <YourApp />
    </ErrorBoundary>
  );
}

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div role="alert">
      <h2>Something went wrong</h2>
      <pre>{error.message}</pre>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  );
}
```

### CSV Streaming (Backend)

Replace memory-based CSV parsing with streaming:

```typescript
import { parse } from "csv-parse";
import { Storage } from "@google-cloud/storage";

// Stream CSV from GCS without loading entire file into memory
const bucket = storage.bucket(GCS_BUCKET_NAME);
const file = bucket.file(gcsPath);

const parser = parse({
  columns: true,
  skip_empty_lines: true,
  trim: true,
  relax_quotes: true, // Handle inconsistent quoting
  relax_column_count: true, // Handle inconsistent column counts
});

file.createReadStream()
  .pipe(parser)
  .on("data", (row) => {
    // Process row by row
    processRow(row);
  })
  .on("error", (err) => {
    console.error("CSV parsing error:", err);
  })
  .on("end", () => {
    console.log("CSV parsing complete");
  });
```

### TypeScript Strict Mode Migration

**Phase 1:** Enable one flag at a time in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "noImplicitAny": true,  // Start here (highest ROI)
    // Add others incrementally:
    // "strictNullChecks": true,
    // "strictFunctionTypes": true,
    // "strictBindCallApply": true,
    // "strictPropertyInitialization": true,
    // "noImplicitThis": true,
    // "alwaysStrict": true,
    // Eventually: "strict": true
  }
}
```

**Phase 2:** Once most files pass, enable full strict mode:

```json
{
  "compilerOptions": {
    "strict": true
  }
}
```

---

## What NOT to Add (Avoid Over-Engineering)

| Technology | Why NOT | What to Do Instead |
|------------|---------|-------------------|
| **express-session** | Adds session store complexity (Redis, database). Overkill for simple API key auth. | Use API key in headers or simple bearer token. Session cookies only if building full user auth system. |
| **passport.js** | Heavy authentication framework. Requires strategy configuration, session management. Not needed for API key auth. | Implement simple middleware: check API key header against env var or Firestore collection. |
| **joi / yup** | Schema validation libraries add dependency weight. express-validator already handles validation. | Stick with express-validator for request validation. |
| **sanitize-filename** | Path traversal prevention library. GCS already handles safe paths. | Use `path.basename()` to strip directory traversal, or let GCS SDK handle it (already safe). |
| **firebase-admin** | This project uses GCP (not Firebase). firebase-admin bundles services not needed (Auth, Realtime DB, FCM). | Continue using individual `@google-cloud/*` packages. More granular, lighter weight. |
| **winston / pino** | Structured logging libraries. GCP Cloud Logging automatically captures console.log. | Use `console.log/error/warn`. Add JSON formatting if needed: `console.log(JSON.stringify({ level, message, metadata }))`. |
| **express-async-handler** | Express 5.2.1 has built-in async error handling. Middleware automatically catches rejected promises. | Use native Express 5 async support. Only add wrapper if downgrading to Express 4. |

---

## Configuration Examples

### Rate Limiting (Per-Endpoint)

```typescript
import rateLimit from "express-rate-limit";

// Upload endpoint: 10 uploads per hour per IP
const uploadLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 10,
  message: "Too many uploads from this IP, please try again later",
});

// Status polling: 100 requests per minute per IP
const statusLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100,
});

app.post("/api/upload", uploadLimiter, upload.single("file"), ...);
app.get("/api/jobs/job-status/:jobId", statusLimiter, ...);
```

### Input Validation Examples

```typescript
import { body, param, validationResult } from "express-validator";

// Upload validation
const validateUpload = [
  body("file").custom((value, { req }) => {
    if (!req.file) throw new Error("No file uploaded");
    if (!req.file.mimetype.includes("csv") && !req.file.originalname.endsWith(".csv")) {
      throw new Error("Only CSV files are allowed");
    }
    if (req.file.size > 50 * 1024 * 1024) {
      throw new Error("File size must be less than 50MB");
    }
    return true;
  }),
];

// Job ID validation
const validateJobId = [
  param("jobId").matches(/^job_[0-9]+_[a-z0-9]+$/).withMessage("Invalid job ID format"),
];

// Apply to routes
app.post("/api/upload", upload.single("file"), validateUpload, async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  // ... rest of upload logic
});
```

### Environment Variable Validation (Startup)

```typescript
// server/index.ts (at top, before creating server)
function validateEnv() {
  const required = [
    "GCS_BUCKET_NAME",
    "GOOGLE_CLOUD_PROJECT",
    "PUBSUB_TOPIC_NAME",
  ];

  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    console.error(`❌ Missing required environment variables: ${missing.join(", ")}`);
    process.exit(1);
  }

  console.log("✅ Environment variables validated");
}

validateEnv();
```

### CORS Production Configuration

```typescript
import cors from "cors";

// Development: allow localhost
const devOrigins = ["http://localhost:5173", "http://localhost:3000"];

// Production: specify actual frontend domain
const prodOrigins = [process.env.FRONTEND_URL];

const corsOptions = {
  origin: process.env.NODE_ENV === "production" ? prodOrigins : devOrigins,
  credentials: true, // Allow cookies if using session auth
  methods: ["GET", "POST", "PUT", "DELETE"],
  allowedHeaders: ["Content-Type", "Authorization"],
};

app.use(cors(corsOptions));
```

### Helmet Production Configuration

```typescript
import helmet from "helmet";

// Basic (recommended starting point)
app.use(helmet());

// Custom CSP for Google Cloud Storage (if serving files from GCS)
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      imgSrc: ["'self'", "https://storage.googleapis.com"],
      scriptSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"], // Tailwind requires unsafe-inline
    },
  },
}));
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **Rate Limiting** | express-rate-limit | express-slow-down | Slow-down is complementary (delays, not blocks). Use both if needed, but rate-limit covers most cases. |
| **Rate Limiting** | express-rate-limit | rate-limiter-flexible | More features (Redis, advanced algorithms), but heavier. Overkill for v1.0. |
| **CSV Parsing (Backend)** | csv-parse | PapaParse | PapaParse is browser-focused. csv-parse is Node.js-native, better streaming. |
| **CSV Parsing (Frontend)** | papaparse / react-papaparse | csv-parser | csv-parser is server-side only. PapaParse works in browser. |
| **Validation** | express-validator | joi | joi requires separate schema files. express-validator integrates inline with routes. |
| **Validation** | express-validator | yup | yup is React-focused (form validation). express-validator is Express-native. |
| **Error Boundaries** | react-error-boundary | Manual class components | Class components are deprecated pattern. Functional approach with hooks is 2026 standard. |
| **Authentication** | Simple API key middleware | passport.js | passport.js is designed for OAuth, local auth strategies. Too heavy for API key validation. |
| **Authentication** | Simple API key middleware | express-session | Session management requires store (Redis/DB). API keys are stateless, simpler for ML job API. |

---

## Security Best Practices (2026)

### API Key Authentication (Simple Approach)

For this use case (ML job submission API), API keys are appropriate:

```typescript
// Middleware: Check API key from header
function requireApiKey(req, res, next) {
  const apiKey = req.headers["x-api-key"];
  const validKeys = process.env.API_KEYS?.split(",") || [];

  if (!apiKey || !validKeys.includes(apiKey)) {
    return res.status(401).json({ error: "Invalid or missing API key" });
  }

  next();
}

// Apply to protected routes
app.post("/api/upload", requireApiKey, upload.single("file"), ...);
```

**Why API Keys Over Sessions:**
- No session store needed (Redis/database)
- Stateless (easier to scale horizontally)
- Simple for service-to-service or CLI usage
- User-facing web UI can still work (store API key in localStorage after initial auth)

**Security Enhancements:**
- Store hashed API keys in Firestore (like passwords)
- Rotate keys periodically
- Associate keys with user/project IDs for tracking
- Add rate limiting per API key (not just per IP)

### Path Traversal Prevention

**Current code is SAFE** because GCS SDK handles paths safely. However, for defense in depth:

```typescript
import path from "path";

// Option 1: Use basename to strip directory components
const safeName = path.basename(req.file.originalname);
const safeFilename = `uploads/${uniqueId}/${safeName}`;

// Option 2: Validate no traversal characters
if (req.file.originalname.includes("..") || req.file.originalname.includes("/")) {
  return res.status(400).json({ error: "Invalid filename" });
}
```

**Do NOT use:** sanitize-filename library (unnecessary dependency when simple validation works).

---

## Migration Checklist

### Phase 1: Security Foundations (High Priority)

- [ ] Install helmet, express-rate-limit, express-validator
- [ ] Apply helmet() as first middleware
- [ ] Replace `cors()` with origin allowlist
- [ ] Add rate limiting to upload and polling endpoints
- [ ] Add input validation to all POST/PUT endpoints
- [ ] Add environment variable validation at startup
- [ ] Implement API key authentication (or defer to v1.1)

### Phase 2: Error Handling & Validation (High Priority)

- [ ] Install react-error-boundary
- [ ] Wrap App component with ErrorBoundary
- [ ] Add error logging (console or monitoring service)
- [ ] Test error boundary with intentional errors
- [ ] Add job ID validation (regex check)
- [ ] Add file type validation (CSV only)
- [ ] Add file size validation (already in multer config)

### Phase 3: CSV Streaming (Medium Priority)

- [ ] Install csv-parse (backend)
- [ ] Replace any memory-based CSV loading with streams
- [ ] Install papaparse (frontend)
- [ ] Add CSV preview before upload (first 10 rows)
- [ ] Add CSV error detection (encoding, malformed)
- [ ] Test with large files (>100MB)

### Phase 4: TypeScript Strict Mode (Low Priority - Incremental)

- [ ] Enable noImplicitAny in tsconfig.json
- [ ] Fix any errors (or add `// @ts-expect-error` comments)
- [ ] Enable strictNullChecks
- [ ] Fix null/undefined errors
- [ ] Enable full strict mode once most errors resolved
- [ ] (Optional) Install typescript-strict-plugin for per-file control

---

## Version Verification Notes

**Confidence Levels:**

| Package | Version | Source | Confidence |
|---------|---------|--------|------------|
| express-rate-limit | 8.3.1 | WebSearch (npm) | MEDIUM (npm blocked, verified via search) |
| express-validator | 7.3.1 | WebSearch (npm) | MEDIUM (npm blocked, verified via search) |
| helmet | 5.x+ | WebSearch (official docs) | MEDIUM (exact version not verified) |
| csv-parse | 5.x+ | WebSearch (official docs) | MEDIUM (part of node-csv ecosystem) |
| react-error-boundary | 6.1.1 | WebSearch (npm) | HIGH (recent release date: Feb 13, 2026) |
| papaparse | 5.x+ | WebSearch | MEDIUM (widely used, version stable) |

**Verification Method:** npm registry was blocked (403 errors), so versions were verified through official documentation, GitHub releases, and recent web searches from 2026. All packages are actively maintained with recent updates.

**Action:** Run `npm info <package>` to verify latest versions before installation.

---

## Sources

### Rate Limiting
- [express-rate-limit npm](https://www.npmjs.com/package/express-rate-limit)
- [How to Add Rate Limiting to Express APIs (2026-02-02)](https://oneuptime.com/blog/post/2026-02-02-express-rate-limiting/view)
- [Rate Limiting in Express.js | Better Stack](https://betterstack.com/community/guides/scaling-nodejs/rate-limiting-express/)

### Input Validation
- [How to Add Input Validation with express-validator (2026-02-02)](https://oneuptime.com/blog/post/2026-02-02-express-validator-input-validation/view)
- [express-validator official docs](https://express-validator.github.io/docs/)

### CSV Streaming
- [CSV Parse - Stream API (official)](https://csv.js.org/parse/api/stream/)
- [Parse Large CSV files via stream in Node.js](https://medium.com/@ayushpratap2494/parse-large-csv-files-via-stream-in-node-js-91c329ff3620)
- [Papa Parse Documentation](https://www.papaparse.com/)

### Error Boundaries
- [How to Implement React Error Boundaries (2026-02-20)](https://oneuptime.com/blog/post/2026-02-20-react-error-boundaries/view)
- [Error Boundaries in React: The Safety Net Every Production App Needs (2026-02)](https://saraswathi-mac.medium.com/error-boundaries-in-react-the-safety-net-every-production-app-needs-f85809bd5563)
- [react-error-boundary npm](https://www.npmjs.com/package/react-error-boundary)

### TypeScript Strict Mode
- [How to Enable TypeScript Strict Mode Effectively (2026-02-20)](https://oneuptime.com/blog/post/2026-02-20-typescript-strict-mode-guide/view)
- [TypeScript Strict Mode Won — Here's How to Use It Right (2026)](https://www.mariorafaelayala.com/blog/typescript-strict-mode-2026)
- [typescript-strict-plugin GitHub](https://github.com/allegro/typescript-strict-plugin)

### Security
- [How to Use Helmet for Security in Express.js (2026-01-25)](https://oneuptime.com/blog/post/2026-01-25-helmet-security-expressjs/view)
- [Express CORS Production Configuration](https://article.arunangshudas.com/how-would-you-manage-cors-in-a-production-express-js-application-45a1138dd6df)
- [Node.js Path Traversal Guide: Examples and Prevention](https://www.stackhawk.com/blog/node-js-path-traversal-guide-examples-and-prevention/)

### Authentication
- [JWT vs Session-Based Authentication (2026-02-20)](https://oneuptime.com/blog/post/2026-02-20-jwt-vs-session-authentication/view)
- [API Authentication Done Right: JWTs, API Keys, and OAuth2 in Production (2026)](https://dev.to/young_gao/api-authentication-done-right-jwts-api-keys-and-oauth2-in-production-38a6)

### Express Session (for reference)
- [Express session middleware (official)](https://expressjs.com/en/resources/middleware/session.html)
- [How to Handle Sessions and Cookies in Express (2026-02-02)](https://oneuptime.com/blog/post/2026-02-02-express-sessions-cookies/view)

### Firebase/GCP
- [Firebase Admin Node.js SDK Release Notes](https://firebase.google.com/support/release-notes/admin/node)

### Async Error Handling
- [Express error handling (official)](https://expressjs.com/en/guide/error-handling.html)
- [How to Implement Error Handling in Express (2026-01-26)](https://oneuptime.com/blog/post/2026-01-26-express-error-handling/view)

---

**Last Updated:** 2026-03-24
**Next Review:** Before v1.0 deployment (verify all package versions with `npm info`)
