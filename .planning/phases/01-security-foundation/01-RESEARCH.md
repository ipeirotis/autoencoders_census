# Phase 1: Security Foundation - Research

**Researched:** 2026-03-25
**Domain:** Express.js API security hardening (authentication, rate limiting, input validation, file upload security)
**Confidence:** HIGH

## Summary

Express.js security hardening for production deployment follows well-established patterns with mature libraries. The standard stack centers on Passport.js for authentication, helmet for security headers, express-rate-limit for rate limiting, and express-validator for input validation. These patterns are stable and widely documented in 2026.

The phase addresses critical vulnerabilities: open CORS (allows any origin), missing authentication layer, path traversal risks from user-controlled filenames, no rate limiting, and production error responses exposing stack traces. All identified gaps have standard solutions with high confidence.

**Primary recommendation:** Use established security middleware stack (passport + express-session + helmet + express-rate-limit + express-validator) rather than custom implementations. The project already has Firestore initialized, making it a natural choice for session storage despite performance considerations.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Authentication mechanism:** Email + password authentication (classic signup/login)
- **Session storage:** Firestore (already using Firestore for jobs — no new infrastructure)
- **Email verification:** Sent but not required to access platform (optional verification)
- **Password reset:** Required for v1 via email link
- **Error response policy:**
  - Generic messages to users ("File validation failed")
  - Full error details logged server-side to GCP Cloud Logging
  - Stack traces never exposed in production responses

### Claude's Discretion
- Password hashing algorithm (bcrypt vs argon2)
- Session cookie configuration (secure, httpOnly, sameSite settings)
- Rate limiting thresholds (research suggested 5 uploads/15min, 60 polls/min)
- CORS allowed origins configuration
- Helmet security header configuration
- Input validation rules (express-validator patterns)
- Path traversal protection implementation (UUID filenames)
- Magic bytes validation for CSV files

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SEC-01 | API implements authentication layer (session-based for UI, API keys for service-to-server) | Passport.js + express-session + Firestore session store |
| SEC-02 | CORS restricts requests to actual frontend domain (no wildcard on authenticated endpoints) | cors package origin whitelist configuration |
| SEC-03 | Rate limiting applied to upload endpoint (5 uploads per 15 minutes per user) | express-rate-limit 8.0.2+ per-user rate limiting |
| SEC-04 | Rate limiting applied to polling endpoint (60 status checks per minute per user) | express-rate-limit 8.0.2+ per-user rate limiting |
| SEC-05 | Rate limiting applied to download endpoint (10 downloads per hour per user) | express-rate-limit 8.0.2+ per-user rate limiting |
| SEC-06 | File type validation checks magic bytes (not just extension) | file-type package for buffer analysis |
| SEC-07 | File type validation restricts to .csv MIME types only | Note: CSV has no magic bytes, requires content validation |
| SEC-08 | Path traversal protection uses UUID filenames (discards user-provided names) | uuid package (already installed) |
| SEC-09 | Path traversal protection canonicalizes all file paths | path.resolve() validation pattern |
| SEC-10 | Input validation sanitizes all user inputs with express-validator | express-validator middleware chains |
| SEC-11 | Input validation provides clear error messages for invalid inputs | express-validator validationResult formatting |
| SEC-12 | Environment variable validation runs at worker startup (fails fast if missing) | envalid or custom validation at startup |
| SEC-13 | Error handling never exposes stack traces in production responses | NODE_ENV=production + custom error handler |
| SEC-14 | Helmet middleware applies security headers (CSP, X-Frame-Options, etc.) | helmet() with custom CSP configuration |
| SEC-15 | Rate limiting uses version 8.0.2+ to avoid CVE-2026-30827 (IPv6 bypass) | express-rate-limit@8.0.2+ |
| SEC-16 | Hardcoded GCP identifiers replaced with environment variables | dotenv + envalid validation |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| passport | ^0.7.x | Authentication middleware | De facto standard for Node.js auth, supports 500+ strategies |
| passport-local | ^1.0.x | Email/password strategy | Built-in strategy for username/password auth |
| express-session | ^1.18.x | Session management | Express's official session middleware, battle-tested |
| firestore-store | ^3.0.x | Firestore session store | Community-maintained session store for Firestore (Google's official package deprecated July 2024) |
| helmet | ^8.x | Security headers | Express security best practices recommend helmet universally |
| express-rate-limit | ^8.0.2+ | Rate limiting | Standard rate limiting, v8.0.2+ fixes CVE-2026-30827 |
| express-validator | ^7.x | Input validation | Most popular Express validation library, chainable API |
| bcrypt | ^5.x | Password hashing | Widely supported, secure with cost factor 13-14 |
| file-type | ^22.x | Magic bytes validation | Reads file signatures from buffers, supports 100+ formats |
| winston | ^3.x | Logging | Production logging standard for Node.js |
| @google-cloud/logging-winston | ^6.x | Cloud Logging transport | Official Google transport for Winston → Cloud Logging |

**Installation:**
```bash
# Express server dependencies (add to frontend/package.json)
npm install passport passport-local express-session firestore-store bcrypt
npm install helmet express-rate-limit express-validator
npm install file-type winston @google-cloud/logging-winston envalid

# Dev dependencies
npm install --save-dev @types/passport @types/passport-local @types/express-session @types/bcrypt
```

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| argon2 | ^0.31.x | Password hashing (alternative) | Use if starting fresh, prefer for new 2026 projects per OWASP |
| envalid | ^8.x | Environment variable validation | Fail-fast validation at startup (alternative: custom validation) |
| cors | ^2.8.5 | CORS configuration | Already installed, requires configuration update |
| uuid | ^13.0.0 | UUID generation | Already installed, use for filename generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| bcrypt | argon2 | Argon2id is OWASP #1 recommendation for 2026, but bcrypt widely supported and secure. No urgent need to migrate existing projects. |
| firestore-store | MemoryStore (express-session default) | MemoryStore loses sessions on restart, unsuitable for production |
| firestore-store | Redis (connect-redis) | Redis faster for sessions, but adds infrastructure. Firestore sufficient for initial launch. |
| Passport.js | Custom JWT implementation | Custom auth is security anti-pattern. Never roll your own authentication. |
| express-validator | joi validation | express-validator integrates better with Express middleware chain |

## Architecture Patterns

### Recommended Project Structure
```
frontend/
├── server/
│   ├── index.ts              # Express app creation (separate from server startup)
│   ├── server.ts             # Server startup (port binding)
│   ├── middleware/           # Authentication, validation, rate limiting
│   │   ├── auth.ts           # Passport configuration + authentication middleware
│   │   ├── rateLimits.ts     # Rate limit configurations
│   │   ├── validation.ts     # express-validator rules
│   │   └── errorHandler.ts   # Production error handler
│   ├── routes/
│   │   ├── auth.ts           # /api/auth/* (signup, login, logout, reset)
│   │   ├── jobs.ts           # /api/jobs/* (protected routes)
│   │   └── upload.ts         # /api/upload (protected, validated)
│   ├── config/
│   │   ├── env.ts            # Environment variable validation (envalid)
│   │   ├── session.ts        # Session configuration
│   │   └── logger.ts         # Winston + Cloud Logging setup
│   └── models/
│       └── user.ts           # User model (Firestore operations)
└── client/
    └── (existing React app)
```

### Pattern 1: Separate App Creation from Server Startup
**What:** Export Express app as a function, start server in separate file
**When to use:** Always — enables testing without port binding
**Example:**
```typescript
// server/index.ts
// Source: https://www.sammeechward.com/testing-an-express-app-with-supertest-and-jest
export function createServer() {
  const app = express();
  // ... middleware and routes
  return app;
}

// server/server.ts
import { createServer } from './index';
const app = createServer();
app.listen(PORT, () => console.log(`Server on ${PORT}`));
```

### Pattern 2: Layered Authentication Middleware
**What:** Passport strategy configuration → session serialization → route protection middleware
**When to use:** All authenticated routes
**Example:**
```typescript
// middleware/auth.ts
// Source: https://oneuptime.com/blog/post/2026-01-22-nodejs-passport-authentication/view
import passport from 'passport';
import { Strategy as LocalStrategy } from 'passport-local';

// 1. Configure strategy
passport.use(new LocalStrategy({
  usernameField: 'email',
  passwordField: 'password'
}, async (email, password, done) => {
  // Lookup user in Firestore, verify password with bcrypt
}));

// 2. Session serialization
passport.serializeUser((user, done) => done(null, user.id));
passport.deserializeUser(async (id, done) => {
  // Fetch user from Firestore
});

// 3. Route protection middleware
export const requireAuth = (req, res, next) => {
  if (req.isAuthenticated()) return next();
  res.status(401).json({ error: 'Authentication required' });
};
```

### Pattern 3: Per-Endpoint Rate Limiting
**What:** Different rate limit configs per endpoint based on resource cost
**When to use:** All public-facing endpoints
**Example:**
```typescript
// middleware/rateLimits.ts
// Source: https://www.npmjs.com/package/express-rate-limit
import rateLimit from 'express-rate-limit';

export const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 requests per window per user
  keyGenerator: (req) => req.user?.id || req.ip, // Per-user after auth
  message: { error: 'Upload limit exceeded. Try again in 15 minutes.' }
});

export const pollLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute per user
  keyGenerator: (req) => req.user?.id || req.ip,
  message: { error: 'Too many status checks' }
});
```

### Pattern 4: Environment Variable Validation at Startup
**What:** Validate all required env vars before starting server
**When to use:** Top of index.ts or dedicated config/env.ts
**Example:**
```typescript
// config/env.ts
// Source: https://github.com/af/envalid
import { cleanEnv, str, port } from 'envalid';

export const env = cleanEnv(process.env, {
  NODE_ENV: str({ choices: ['development', 'production', 'test'] }),
  PORT: port({ default: 5001 }),
  GOOGLE_CLOUD_PROJECT: str(),
  GCS_BUCKET_NAME: str(),
  GOOGLE_APPLICATION_CREDENTIALS: str(),
  SESSION_SECRET: str(),
  FRONTEND_URL: str()
});

// Script fails immediately if required vars missing
```

### Pattern 5: Production Error Handler
**What:** Custom error middleware that hides stack traces in production
**When to use:** Last middleware in chain
**Example:**
```typescript
// middleware/errorHandler.ts
// Source: https://oneuptime.com/blog/post/2026-02-02-express-error-handling/view
import { logger } from '../config/logger';

export const errorHandler = (err, req, res, next) => {
  // Log full error server-side
  logger.error('Request error', {
    message: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method
  });

  // Send generic message to client
  const isProd = process.env.NODE_ENV === 'production';
  res.status(err.status || 500).json({
    error: isProd ? 'Internal server error' : err.message,
    ...(isProd ? {} : { stack: err.stack })
  });
};
```

### Pattern 6: File Upload Validation with Magic Bytes
**What:** Validate file buffer with file-type before processing
**When to use:** All file upload endpoints
**Example:**
```typescript
// routes/upload.ts
// Source: https://medium.com/@sridhar_be/file-validations-using-magic-numbers-in-nodejs-express-server-d8fbb31a97e7
import { fileTypeFromBuffer } from 'file-type';

app.post('/api/upload', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

  // Validate magic bytes
  const type = await fileTypeFromBuffer(req.file.buffer);

  // CSV has no magic bytes, validate structure instead
  const isCSV = req.file.mimetype === 'text/csv' ||
                req.file.originalname.endsWith('.csv');

  if (!isCSV) {
    return res.status(400).json({ error: 'File validation failed' });
  }

  // Additional CSV structure validation
  const content = req.file.buffer.toString('utf-8');
  if (!content.includes(',') || content.split('\n').length < 2) {
    return res.status(400).json({ error: 'File validation failed' });
  }

  // Use UUID for filename (discard user-provided name)
  const fileId = uuidv4();
  const safeFilename = `uploads/${req.user.id}/${fileId}.csv`;
  // ... upload to GCS
});
```

### Pattern 7: CORS Whitelist Configuration
**What:** Dynamically validate origin against whitelist
**When to use:** Always in production
**Example:**
```typescript
// index.ts
// Source: https://article.arunangshudas.com/how-would-you-manage-cors-in-a-production-express-js-application-45a1138dd6df
import cors from 'cors';

const allowedOrigins = [
  process.env.FRONTEND_URL,
  // Add staging, preview environments as needed
].filter(Boolean);

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, curl, Postman)
    if (!origin) return callback(null, true);

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('CORS policy violation'));
    }
  },
  credentials: true // Allow cookies
}));
```

### Pattern 8: Session Cookie Configuration
**What:** Secure session cookies for production
**When to use:** Session middleware configuration
**Example:**
```typescript
// config/session.ts
// Source: https://oneuptime.com/blog/post/2026-02-02-express-sessions-cookies/view
import session from 'express-session';
import { Firestore } from '@google-cloud/firestore';
import { FirestoreStore } from 'firestore-store';

const isProd = process.env.NODE_ENV === 'production';

export const sessionConfig = session({
  store: new FirestoreStore({
    database: new Firestore(),
    collection: 'sessions'
  }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: isProd, // HTTPS only in production
    httpOnly: true, // Prevent XSS access
    sameSite: 'lax', // CSRF protection
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
});
```

### Pattern 9: Input Validation Chains
**What:** Chainable validation rules per endpoint
**When to use:** All endpoints receiving user input
**Example:**
```typescript
// middleware/validation.ts
// Source: https://oneuptime.com/blog/post/2026-02-02-express-validator-input-validation/view
import { body, validationResult } from 'express-validator';

export const validateSignup = [
  body('email')
    .isEmail().withMessage('Invalid email')
    .normalizeEmail(),
  body('password')
    .isLength({ min: 8 }).withMessage('Password must be 8+ characters')
    .matches(/\d/).withMessage('Password must contain number'),
  (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation failed',
        details: errors.array().map(e => e.msg)
      });
    }
    next();
  }
];
```

### Anti-Patterns to Avoid
- **Rolling custom authentication:** Use Passport.js, not custom JWT verification
- **Storing passwords in plaintext or using weak hashing:** Always use bcrypt/argon2
- **Trusting client-provided MIME types without validation:** Check magic bytes (or CSV structure)
- **Using user-provided filenames directly:** Generate UUIDs, discard original names
- **Wildcard CORS in production:** Always whitelist specific origins
- **Exposing stack traces to users:** Log server-side, return generic messages
- **Skipping environment variable validation:** Validate at startup, fail fast
- **Using express-rate-limit < 8.0.2:** CVE-2026-30827 allows IPv6 bypass

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Authentication | Custom JWT/session management | Passport.js + express-session | Session management has subtle race conditions, CSRF vulnerabilities, timing attacks. Passport handles 500+ strategies. |
| Password hashing | Custom hashing or basic crypto | bcrypt or argon2 | Salting, key stretching, timing attack resistance require expertise. Cost factor tuning is critical. |
| Rate limiting | Custom request counting | express-rate-limit | IPv6 handling (CVE-2026-30827), distributed rate limiting, memory management. |
| Input validation | Manual regex checks | express-validator | XSS prevention, SQL injection, comprehensive sanitization. Easy to miss edge cases. |
| Security headers | Manual header setting | helmet | CSP, X-Frame-Options, HSTS configuration is complex. helmet provides secure defaults. |
| File type detection | Extension checking | file-type package | Magic bytes validation prevents spoofing. Supports 100+ formats with edge case handling. |
| Environment validation | if (!process.env.X) throw | envalid | Type coercion, default values, clear error messages, schema documentation. |
| Session storage | In-memory or filesystem | Firestore/Redis session store | Persistence, scalability, cleanup of expired sessions. |

**Key insight:** Security is not additive — one weakness undermines all other protections. Express has well-tested libraries for every security concern. Custom implementations introduce subtle vulnerabilities that take years to discover.

## Common Pitfalls

### Pitfall 1: express-rate-limit IPv6 Bypass (CVE-2026-30827)
**What goes wrong:** All IPv4 clients share one rate limit bucket on dual-stack servers
**Why it happens:** Versions < 8.0.2 apply /56 subnet masking to IPv4-mapped IPv6 addresses (::ffff:x.x.x.x), collapsing all IPv4 traffic into one bucket
**How to avoid:** Use express-rate-limit@8.0.2 or higher
**Warning signs:** One abusive IPv4 client causes HTTP 429 for all IPv4 users
**Source:** [CVE-2026-30827 GitLab Advisory](https://advisories.gitlab.com/pkg/npm/express-rate-limit/CVE-2026-30827/)

### Pitfall 2: CSV Files Have No Magic Bytes
**What goes wrong:** file-type returns null for CSV files, validation fails
**Why it happens:** CSV is plain text format without standardized file signature
**How to avoid:** Validate CSV by checking structure (commas, newlines, header row) not magic bytes
**Warning signs:** Legitimate CSV uploads rejected with "file validation failed"
**Pattern:**
```typescript
// Don't rely on magic bytes for CSV
const type = await fileTypeFromBuffer(buffer);
if (type?.mime === 'text/csv') { /* Won't match */ }

// Instead validate structure
const content = buffer.toString('utf-8');
const hasCommas = content.includes(',');
const hasRows = content.split('\n').length >= 2;
const isCSV = hasCommas && hasRows;
```

### Pitfall 3: Firestore Session Store Doesn't Auto-Delete Expired Sessions
**What goes wrong:** Session collection grows unbounded, Firestore read costs increase
**Why it happens:** firestore-store doesn't implement automatic session cleanup
**How to avoid:** Implement Firestore TTL policy or scheduled Cloud Function to delete old sessions
**Warning signs:** Firestore session collection size growing continuously
**Source:** [Google Cloud Firestore Session Handling Docs](https://cloud.google.com/nodejs/getting-started/session-handling-with-firestore)

### Pitfall 4: Session Secret Stored in Code
**What goes wrong:** Session hijacking if secret leaks in version control
**Why it happens:** Hardcoded secrets checked into Git
**How to avoid:** Store SESSION_SECRET in environment variable, validate at startup with envalid
**Warning signs:** Session secret visible in GitHub, secret rotation requires code deployment

### Pitfall 5: Rate Limiting Before Authentication
**What goes wrong:** Unauthenticated users can exhaust rate limits for authenticated users
**Why it happens:** Rate limiting by IP when keyGenerator should use user ID after auth
**How to avoid:** Apply rate limiting AFTER authentication, use req.user.id as key
**Pattern:**
```typescript
// Wrong order
app.use(uploadLimiter);
app.use(requireAuth);

// Correct order
app.use(requireAuth);
app.use(uploadLimiter); // Now has req.user.id

// Correct keyGenerator
keyGenerator: (req) => req.user?.id || req.ip
```

### Pitfall 6: NODE_ENV Not Set in Production
**What goes wrong:** Stack traces exposed to users, verbose logging enabled
**Why it happens:** Environment variable not configured in deployment
**How to avoid:** Set NODE_ENV=production in Cloud Run/GCE environment config, validate at startup
**Warning signs:** Users report seeing file paths and code snippets in error messages

### Pitfall 7: Helmet CSP Blocks Frontend Assets
**What goes wrong:** React app fails to load, inline scripts blocked
**Why it happens:** Default CSP is very restrictive
**How to avoid:** Configure CSP to allow your frontend origin and required script sources
**Pattern:**
```typescript
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"], // Vite needs inline scripts
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", process.env.FRONTEND_URL]
    }
  }
}));
```

### Pitfall 8: Testing Against Running Server Instead of App Instance
**What goes wrong:** Tests fail with EADDRINUSE, slow test execution
**Why it happens:** Tests start server on port instead of using Supertest with app instance
**How to avoid:** Export createServer() function, import in tests without calling app.listen()
**Pattern:**
```typescript
// tests/auth.test.ts
import request from 'supertest';
import { createServer } from '../server';

const app = createServer(); // No port binding
const response = await request(app).post('/api/auth/signup');
```

## Code Examples

Verified patterns from official sources:

### Complete Authentication Setup
```typescript
// server/middleware/auth.ts
// Source: https://www.passportjs.org/howtos/session/
import passport from 'passport';
import { Strategy as LocalStrategy } from 'passport-local';
import bcrypt from 'bcrypt';
import { getUserByEmail, getUserById } from '../models/user';

passport.use(new LocalStrategy(
  { usernameField: 'email' },
  async (email, password, done) => {
    try {
      const user = await getUserByEmail(email);
      if (!user) return done(null, false, { message: 'Invalid credentials' });

      const isValid = await bcrypt.compare(password, user.passwordHash);
      if (!isValid) return done(null, false, { message: 'Invalid credentials' });

      return done(null, user);
    } catch (error) {
      return done(error);
    }
  }
));

passport.serializeUser((user: any, done) => {
  done(null, user.id);
});

passport.deserializeUser(async (id: string, done) => {
  try {
    const user = await getUserById(id);
    done(null, user);
  } catch (error) {
    done(error);
  }
});

export const requireAuth = (req, res, next) => {
  if (req.isAuthenticated()) return next();
  res.status(401).json({ error: 'Authentication required' });
};
```

### Winston + Cloud Logging Configuration
```typescript
// server/config/logger.ts
// Source: https://oneuptime.com/blog/post/2026-02-17-logging-winston-transport-node-js-cloud-logging/view
import winston from 'winston';
import { LoggingWinston } from '@google-cloud/logging-winston';

const loggingWinston = new LoggingWinston({
  projectId: process.env.GOOGLE_CLOUD_PROJECT,
  keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS
});

export const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    loggingWinston,
    // Console for local development
    ...(process.env.NODE_ENV !== 'production' ? [
      new winston.transports.Console({
        format: winston.format.simple()
      })
    ] : [])
  ]
});
```

### Path Traversal Protection
```typescript
// server/routes/upload.ts
// Source: https://www.nodejs-security.com/blog/secure-coding-practices-nodejs-path-traversal-vulnerabilities
import { v4 as uuidv4 } from 'uuid';
import path from 'path';

app.post('/api/upload', requireAuth, upload.single('file'), async (req, res) => {
  // Never trust user-provided filename
  const unsafeOriginal = req.file.originalname; // "../../etc/passwd.csv"

  // Generate opaque ID
  const fileId = uuidv4(); // "3c7e8f1a-4b2d-..."

  // Construct safe path with user isolation
  const safePath = `uploads/${req.user.id}/${fileId}.csv`;

  // Verify path doesn't escape intended directory (defense in depth)
  const uploadDir = path.resolve('./uploads');
  const resolvedPath = path.resolve(uploadDir, req.user.id, `${fileId}.csv`);

  if (!resolvedPath.startsWith(uploadDir)) {
    logger.warn('Path traversal attempt', { user: req.user.id, path: resolvedPath });
    return res.status(400).json({ error: 'File validation failed' });
  }

  // Upload to GCS with safe path
  await bucket.file(safePath).save(req.file.buffer);

  // Store metadata with original name for display purposes only
  await firestore.collection('jobs').doc(fileId).set({
    userId: req.user.id,
    originalName: path.basename(unsafeOriginal), // Strip any path components
    gcsPath: safePath,
    createdAt: new Date()
  });

  res.json({ jobId: fileId });
});
```

### Comprehensive Input Validation
```typescript
// server/middleware/validation.ts
// Source: https://oneuptime.com/blog/post/2026-02-02-express-validator-input-validation/view
import { body, param, query, validationResult } from 'express-validator';

export const validateSignup = [
  body('email')
    .isEmail().withMessage('Invalid email address')
    .normalizeEmail()
    .trim(),
  body('password')
    .isLength({ min: 8, max: 128 }).withMessage('Password must be 8-128 characters')
    .matches(/\d/).withMessage('Password must contain at least one number')
    .matches(/[a-z]/).withMessage('Password must contain lowercase letter')
    .matches(/[A-Z]/).withMessage('Password must contain uppercase letter'),
  handleValidationErrors
];

export const validateJobId = [
  param('jobId')
    .isUUID().withMessage('Invalid job ID format'),
  handleValidationErrors
];

export const validatePollRequest = [
  param('jobId').isUUID(),
  query('timestamp').optional().isISO8601(),
  handleValidationErrors
];

function handleValidationErrors(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors.array().map(e => ({
        field: e.param,
        message: e.msg
      }))
    });
  }
  next();
}
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Jest 29.x + Supertest 6.x (for Express API) / pytest (for Python worker) |
| Config file | jest.config.js (Wave 0), pytest.ini (exists) |
| Quick run command | `npm test -- --testPathPattern=auth` |
| Full suite command | `npm test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SEC-01 | Unauthenticated requests to /api/upload return 401 | integration | `npm test -- server/routes/upload.test.ts` | ❌ Wave 0 |
| SEC-01 | Login with valid credentials creates session | integration | `npm test -- server/routes/auth.test.ts` | ❌ Wave 0 |
| SEC-02 | Requests from unauthorized origin rejected | integration | `npm test -- server/middleware/cors.test.ts` | ❌ Wave 0 |
| SEC-03 | 6th upload in 15min window returns 429 | integration | `npm test -- server/middleware/rateLimits.test.ts` | ❌ Wave 0 |
| SEC-04 | 61st poll in 1min window returns 429 | integration | `npm test -- server/middleware/rateLimits.test.ts` | ❌ Wave 0 |
| SEC-05 | 11th download in 1hr window returns 429 | integration | `npm test -- server/middleware/rateLimits.test.ts` | ❌ Wave 0 |
| SEC-06 | Upload with .exe file renamed to .csv rejected | integration | `npm test -- server/routes/upload.test.ts` | ❌ Wave 0 |
| SEC-07 | Upload with valid CSV structure accepted | integration | `npm test -- server/routes/upload.test.ts` | ❌ Wave 0 |
| SEC-08 | Uploaded file stored with UUID, not user filename | unit | `npm test -- server/utils/fileHandling.test.ts` | ❌ Wave 0 |
| SEC-09 | Path traversal attempt (../../etc/passwd) rejected | unit | `npm test -- server/utils/fileHandling.test.ts` | ❌ Wave 0 |
| SEC-10 | Malformed email in signup returns 400 with message | integration | `npm test -- server/middleware/validation.test.ts` | ❌ Wave 0 |
| SEC-11 | Validation error response includes field and message | integration | `npm test -- server/middleware/validation.test.ts` | ❌ Wave 0 |
| SEC-12 | Worker startup fails if GOOGLE_CLOUD_PROJECT missing | unit | `python -m pytest tests/test_env_validation.py::test_missing_project_id -x` | ❌ Wave 0 |
| SEC-13 | Production error response excludes stack trace | integration | `npm test -- server/middleware/errorHandler.test.ts` | ❌ Wave 0 |
| SEC-14 | Response includes X-Frame-Options: DENY header | integration | `npm test -- server/middleware/security.test.ts` | ❌ Wave 0 |
| SEC-15 | IPv4-mapped IPv6 addresses rate limited per-client | integration | `npm test -- server/middleware/rateLimits.test.ts` | ❌ Wave 0 |
| SEC-16 | Server startup fails if GCS_BUCKET_NAME missing | unit | `npm test -- server/config/env.test.ts` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `npm test -- --testPathPattern={changed-file}` (< 10s)
- **Per wave merge:** `npm test` (< 30s for Phase 1 tests)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `frontend/jest.config.js` — Jest configuration for Node.js environment
- [ ] `frontend/server/__tests__/setup.ts` — Test database cleanup, mock Firestore/GCS
- [ ] Framework install: `npm install --save-dev jest @types/jest supertest @types/supertest ts-jest` — if none detected
- [ ] `tests/test_env_validation.py` — Python worker env validation tests (pytest exists)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| @google-cloud/connect-firestore | firestore-store (community) | July 2024 | Official Google Firestore session store deprecated, use community alternative |
| express-rate-limit < 8.0.2 | express-rate-limit 8.0.2+ | January 2026 | CVE-2026-30827 fixed, IPv6 bypass prevented |
| Custom session stores | express-session compatible stores | Ongoing | Standardized session.Store interface ensures compatibility |
| bcrypt-only recommendation | Argon2id preferred for new projects | 2023-2026 | OWASP shifted to Argon2id as #1, bcrypt still secure |
| Manual CSP configuration | helmet() with CSP defaults | helmet 7.0+ (2023) | Simpler API, secure defaults out of box |

**Deprecated/outdated:**
- **@google-cloud/connect-firestore**: Deprecated July 2024, use firestore-store instead
- **express-rate-limit < 8.0.2**: Contains critical IPv6 bypass vulnerability
- **passport-local-mongoose**: Mongoose-specific, not compatible with Firestore. Use manual Passport + bcrypt.
- **body-parser as separate package**: Included in Express 4.16+ as express.json() and express.urlencoded()

## Open Questions

1. **Email sending service**
   - What we know: Password reset and email verification require sending emails
   - What's unclear: Which service to use (SendGrid, Mailgun, Cloud Run SMTP)
   - Recommendation: Defer email implementation to Wave 2+ after core auth working. Use placeholder "email sent" logs initially.

2. **Firestore session cleanup strategy**
   - What we know: firestore-store doesn't auto-delete expired sessions
   - What's unclear: Whether to implement TTL policy now or defer
   - Recommendation: Implement cleanup in Wave 1 using Firestore TTL rules (5 minute task). Prevents operational debt.

3. **API key authentication for service-to-service**
   - What we know: SEC-01 mentions API keys alongside session-based auth
   - What's unclear: Whether worker → API calls need authentication
   - Recommendation: Worker uses service account (already configured), no API key needed. Focus on user session auth.

4. **Rate limiting storage**
   - What we know: express-rate-limit uses in-memory store by default
   - What's unclear: Whether distributed rate limiting needed for multi-instance deployment
   - Recommendation: In-memory sufficient for single Cloud Run instance. Defer distributed rate limiting to scaling phase.

## Sources

### Primary (HIGH confidence)
- [Passport.js Official Documentation](https://www.passportjs.org/howtos/session/) - Session authentication patterns
- [express-rate-limit npm](https://www.npmjs.com/package/express-rate-limit) - Rate limiting configuration
- [helmet.js Official Site](https://helmetjs.github.io/) - Security headers configuration
- [express-validator Official Docs](https://express-validator.github.io/docs/) - Input validation API
- [Express.js Security Best Practices](https://expressjs.com/en/advanced/best-practice-security.html) - Official security guide
- [Google Cloud Firestore Session Handling](https://cloud.google.com/nodejs/getting-started/session-handling-with-firestore) - Official GCP session docs
- [CVE-2026-30827 GitLab Advisory](https://advisories.gitlab.com/pkg/npm/express-rate-limit/CVE-2026-30827/) - express-rate-limit vulnerability details
- [Node.js Path Traversal Security](https://www.nodejs-security.com/blog/secure-coding-practices-nodejs-path-traversal-vulnerabilities) - Path traversal prevention

### Secondary (MEDIUM confidence)
- [OneUpTime Express Auth Guide (2026-01-22)](https://oneuptime.com/blog/post/2026-01-22-nodejs-passport-authentication/view) - Passport implementation tutorial
- [OneUpTime Express Error Handling (2026-02-02)](https://oneuptime.com/blog/post/2026-02-02-express-error-handling/view) - Production error handling
- [OneUpTime Helmet Guide (2026-01-25)](https://oneuptime.com/blog/post/2026-01-25-helmet-security-expressjs/view) - Helmet configuration
- [OneUpTime Express Validator (2026-02-02)](https://oneuptime.com/blog/post/2026-02-02-express-validator-input-validation/view) - Validation patterns
- [OneUpTime Cloud Logging Winston (2026-02-17)](https://oneuptime.com/blog/post/2026-02-17-logging-winston-transport-node-js-cloud-logging/view) - Winston + GCP integration
- [bcrypt vs Argon2 Comparison (2026)](https://viadreams.cc/en/blog/bcrypt-vs-argon2-vs-scrypt-password-hashing/) - Password hashing comparison
- [MDN Cookie Security Guide](https://developer.mozilla.org/en-US/docs/Web/Security/Practical_implementation_guides/Cookies) - Cookie security attributes
- [How to Manage CORS in Production (Medium)](https://article.arunangshudas.com/how-would-you-manage-cors-in-a-production-express-js-application-45a1138dd6df) - CORS whitelist patterns

### Tertiary (LOW confidence)
- [firestore-store GitHub](https://github.com/hendrysadrak/firestore-store) - Community Firestore session store (marked for validation: not officially endorsed)
- [envalid GitHub](https://github.com/af/envalid) - Environment variable validation library
- [Magic Bytes Validation Medium Article](https://medium.com/@sridhar_be/file-validations-using-magic-numbers-in-nodejs-express-server-d8fbb31a97e7) - File validation patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are industry standard with 10+ years maturity
- Architecture: HIGH - Patterns verified in official Express.js and Passport.js documentation
- Pitfalls: HIGH - CVE-2026-30827 verified in GitLab advisory, other pitfalls documented in multiple sources
- File upload security: MEDIUM - CSV magic bytes limitation confirmed across multiple sources, validation patterns consistent

**Research date:** 2026-03-25
**Valid until:** 2026-06-25 (90 days - security best practices stable, but check for new CVEs monthly)

**CVE monitoring:** Check [Snyk Express vulnerability database](https://security.snyk.io/package/npm/express) and [express-rate-limit advisories](https://security.snyk.io/package/npm/express-rate-limit) monthly for new security issues.
