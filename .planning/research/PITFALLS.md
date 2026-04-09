# Domain Pitfalls: Production Hardening ML Web Apps

**Domain:** ML web application (autoencoder outlier detection)
**Researched:** 2026-03-24
**Confidence:** HIGH

This document focuses on pitfalls when **adding** production features (authentication, validation, security, error handling, operational features) to an **existing** Express + React + GCP stack. Not generic security advice—specific mistakes made during hardening.

---

## Critical Pitfalls

Mistakes that cause security breaches, data loss, or complete rewrites.

### Pitfall 1: Authentication Added Without Session Regeneration

**What goes wrong:** Adding authentication middleware (API keys, sessions) but failing to regenerate session IDs on login creates session fixation vulnerabilities. Attacker presets session ID, victim logs in with that ID, attacker hijacks session.

**Why it happens:** express-session provides `.regenerate()` function, but it's the developer's responsibility to call it. Popular libraries like Passport.js don't call this automatically, leaving the vulnerability in place.

**Consequences:**
- Attacker can preset session IDs and hijack accounts after successful login
- Violates security assumptions of newly-added authentication
- In ML apps: attacker gains access to user's uploaded datasets and results

**Prevention:**
```javascript
// WRONG: Just setting user after authentication
app.post('/login', (req, res) => {
  if (validCredentials) {
    req.session.userId = user.id;
    res.json({ success: true });
  }
});

// RIGHT: Regenerate session on login
app.post('/login', (req, res) => {
  if (validCredentials) {
    req.session.regenerate((err) => {
      if (err) return res.status(500).json({ error: 'Login failed' });
      req.session.userId = user.id;
      res.json({ success: true });
    });
  }
});
```

**Detection:**
- Security audit: check all authentication endpoints for `.regenerate()` calls
- Pen test: attempt session fixation attack (preset PHPSESSID/connect.sid cookie)

**Phase mapping:** Address in Phase 1 (authentication implementation)

**Sources:**
- [Session Authentication Best Practices](https://medium.com/@kasongokakumbiguy/here-are-security-best-practices-for-session-management-nodejs-and-express-3257c2799f46)
- [Should you use Express-session for production?](https://supertokens.com/blog/should-you-use-express-session-for-your-production-app)

---

### Pitfall 2: Rate Limiting Bypass via IPv4-Mapped IPv6 Addresses (CVE-2026-30827)

**What goes wrong:** Adding express-rate-limit on dual-stack (IPv4+IPv6) servers causes all IPv4 clients to share a single rate limit bucket. One client hitting the limit causes HTTP 429 for ALL other IPv4 clients.

**Why it happens:** Node.js returns IPv4 requests as IPv4-mapped IPv6 addresses (`::ffff:x.x.x.x`) on dual-stack servers. Default keyGenerator applies /56 subnet masking to all IPv6 addresses, including IPv4-mapped ones. Since first 80 bits of IPv4-mapped addresses are zero, all IPv4 traffic collapses to `::/56` key.

**Consequences:**
- Single malicious user can DoS all legitimate IPv4 users
- Rate limiting completely ineffective for IPv4 traffic
- In ML apps with 10-15min jobs: one user triggering rate limit blocks all other users from uploading

**Prevention:**
```javascript
// WRONG: Default keyGenerator (affected versions 8.0.0-8.0.1, 8.1.0, 8.2.0-8.2.1)
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10
});

// RIGHT: Upgrade to fixed version or custom keyGenerator
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  keyGenerator: (req) => {
    // Extract real IPv4 from IPv4-mapped IPv6
    const ip = req.ip;
    if (ip.startsWith('::ffff:')) {
      return ip.substring(7); // Strip ::ffff: prefix
    }
    return ip;
  }
});
```

**Detection:**
- Check express-rate-limit version: must be ≥8.0.2, ≥8.1.1, or ≥8.2.2
- Test from multiple IPv4 clients simultaneously
- Monitor rate limit bucket keys in logs

**Phase mapping:** Address in Phase 1 (rate limiting implementation)

**Sources:**
- [CVE-2026-30827 Advisory](https://advisories.gitlab.com/pkg/npm/express-rate-limit/CVE-2026-30827/)
- [How to Handle IPv6 in Rate Limiting](https://oneuptime.com/blog/post/2026-03-20-ipv6-in-rate-limiting-middleware/view)

---

### Pitfall 3: CORS Configured with Dynamic Origin Reflection

**What goes wrong:** Adding CORS restrictions but dynamically reflecting the request's `Origin` header back in `Access-Control-Allow-Origin` without validating it against an allowlist. Effectively identical to wildcard CORS—any domain can make authenticated requests.

**Why it happens:** Developers want to allow multiple frontend domains (dev, staging, prod) so they implement dynamic reflection thinking it's more restrictive than `*`. But without allowlist checking, it's equally permissive.

**Consequences:**
- Any malicious site can make authenticated requests to your API
- Bypasses CORS protection entirely
- In ML apps: attacker site can upload files, access results, consume Vertex AI quota via victim's session
- 23% of scanned websites had CORS misconfigurations (2026 data)

**Prevention:**
```javascript
// WRONG: Dynamic reflection without validation
app.use(cors({
  origin: (origin, callback) => {
    callback(null, origin); // Reflects any origin
  },
  credentials: true
}));

// RIGHT: Explicit allowlist
const allowedOrigins = [
  'https://autoencoder.example.com',
  'https://dev.autoencoder.example.com'
];

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, origin);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true
}));
```

**Detection:**
- Send request with arbitrary `Origin: https://evil.com`
- Check if response includes `Access-Control-Allow-Origin: https://evil.com`
- If yes, CORS is bypassed

**Phase mapping:** Address in Phase 1 (CORS hardening)

**Sources:**
- [Express.js CORS Misconfiguration](https://www.sourcery.ai/vulnerabilities/javascript-express-security-cors-misconfiguration)
- [CORS Misconfigurations & Bypass](https://book.hacktricks.xyz/pentesting-web/cors-bypass)
- [Exploiting CORS Misconfiguration](https://www.intigriti.com/researchers/blog/hacking-tools/exploiting-cors-misconfiguration-vulnerabilities)

---

### Pitfall 4: Path Traversal Still Possible After Filename Sanitization

**What goes wrong:** Adding path traversal protection by removing `..` from filenames, but missing other bypasses: URL encoding (`%2e%2e`), path separator variations (`..\/`, `..\`), null bytes (`../file.txt%00.csv`), Unicode variations.

**Why it happens:** Developers sanitize obvious `..` sequences but don't account for encoding variations, platform-specific separators, or null byte injection.

**Consequences:**
- Attacker can read/write outside intended directory
- In GCS context: overwrite system files, access other users' uploads
- CVE-2026-3089 (Actual Sync Server): path traversal via `x-actual-file-id` header with `../`

**Prevention:**
```javascript
// WRONG: Only removes literal ../
function sanitizeFilename(filename) {
  return filename.replace(/\.\./g, '');
}

// RIGHT: Comprehensive sanitization
const path = require('path');

function sanitizeFilename(filename) {
  // Decode URI components first
  let clean = decodeURIComponent(filename);

  // Remove null bytes
  clean = clean.replace(/\0/g, '');

  // Get just the basename (removes any path components)
  clean = path.basename(clean);

  // Remove any remaining path traversal sequences
  clean = clean.replace(/\.\./g, '');

  // Validate it's just a filename
  if (clean.includes('/') || clean.includes('\\')) {
    throw new Error('Invalid filename');
  }

  return clean;
}
```

**Detection:**
- Test with encoded traversal: `%2e%2e%2f`
- Test with null bytes: `../etc/passwd%00.csv`
- Test with mixed separators: `..\/file.txt`
- Check if file appears in unexpected location

**Phase mapping:** Address in Phase 1 (input validation)

**Sources:**
- [CVE-2026-3089: Path Traversal in Actual Sync Server](https://advisories.gitlab.com/pkg/npm/@actual-app/sync-server/CVE-2026-3089/)
- [Bypassing File Upload Restrictions](https://blog.doyensec.com/2025/01/09/cspt-file-upload.html)
- [CSV Upload Security](https://www.eliranturgeman.com/2026/03/14/uploads-attack-surface/)

---

### Pitfall 5: Pub/Sub Ack Race Condition Causing Duplicate Processing

**What goes wrong:** Adding message validation to worker but acknowledging message before processing completes, or after processing fails. On crash/restart, Pub/Sub redelivers message and worker processes job twice, wasting Vertex AI quota and confusing users with duplicate results.

**Why it happens:** Developer acks message immediately upon receipt to prevent redelivery, not realizing Pub/Sub needs ack **after** successful processing. Or worker crashes between processing and ack.

**Consequences:**
- Duplicate training jobs consume 2x Vertex AI quota
- Two sets of results written to Firestore (race condition on status updates)
- User sees "completed" status flip back to "processing" when duplicate runs
- Ack deadline expires during long processing → redelivery even after success

**Prevention:**
```javascript
// WRONG: Ack immediately
subscription.on('message', async (message) => {
  message.ack(); // Acked before processing!
  await processJob(message.data);
});

// RIGHT: Ack only after success, extend deadline for long jobs
subscription.on('message', async (message) => {
  try {
    // Extend ack deadline for long-running Vertex AI jobs
    const extendInterval = setInterval(() => {
      message.modifyAckDeadline(600); // 10 min extension
    }, 300000); // Extend every 5 min

    await processJob(message.data);

    clearInterval(extendInterval);
    message.ack(); // Only ack on success
  } catch (error) {
    clearInterval(extendInterval);
    message.nack(); // Explicit nack for retry
  }
});

// BEST: Make processing idempotent with job ID tracking
const processedJobs = new Set(); // Or use Redis for distributed workers

subscription.on('message', async (message) => {
  const jobId = message.data.jobId;

  if (processedJobs.has(jobId)) {
    message.ack(); // Already processed, just ack
    return;
  }

  try {
    await processJob(message.data);
    processedJobs.add(jobId);
    message.ack();
  } catch (error) {
    message.nack();
  }
});
```

**Detection:**
- Monitor Firestore for jobs with multiple completion timestamps
- Check Vertex AI for duplicate training jobs with same dataset
- Log message IDs and check for duplicates
- Simulate worker crash during processing

**Phase mapping:** Address in Phase 2 (worker reliability)

**Sources:**
- [Pub/Sub Exactly-Once Delivery](https://oneuptime.com/blog/post/2026-01-27-pubsub-exactly-once/view)
- [Pub/Sub Message Deduplication](https://oneuptime.com/blog/post/2026-02-17-how-to-handle-pubsub-message-deduplication-in-subscriber-applications/view)
- [Troubleshooting Pull Subscriptions](https://docs.cloud.google.com/pubsub/docs/pull-troubleshooting)

---

### Pitfall 6: Error Boundary Memory Leak from Polling Intervals

**What goes wrong:** Adding React error boundary to catch rendering errors, but error boundary contains `setInterval` for polling job status that isn't cleared when error boundary unmounts or when error state triggers. Interval keeps running, accumulating memory and making redundant API calls.

**Why it happens:** Error boundary introduces new lifecycle complexity. Developer adds polling in component that gets wrapped by error boundary, but doesn't realize error boundary remounting doesn't trigger cleanup of the errored component's intervals.

**Consequences:**
- Memory leak: intervals pile up with each error/recovery cycle
- API spam: hundreds of polling requests for jobs that already completed
- In ML apps: polling a 15-min Vertex AI job every 2sec for hours after error
- CVE-2026-23864: React Server Component infinite loop caused OOM crashes

**Prevention:**
```javascript
// WRONG: useEffect with interval but missing cleanup
function JobStatusPolling({ jobId }) {
  const [status, setStatus] = useState('pending');

  useEffect(() => {
    const pollInterval = setInterval(async () => {
      const result = await fetch(`/api/job-status/${jobId}`);
      setStatus(await result.json());
    }, 2000);
    // Missing cleanup!
  }, [jobId]);

  return <div>Status: {status}</div>;
}

// RIGHT: Always return cleanup function
function JobStatusPolling({ jobId }) {
  const [status, setStatus] = useState('pending');

  useEffect(() => {
    let mounted = true; // Prevent state updates after unmount

    const pollInterval = setInterval(async () => {
      const result = await fetch(`/api/job-status/${jobId}`);
      const data = await result.json();
      if (mounted) setStatus(data);
    }, 2000);

    return () => {
      mounted = false;
      clearInterval(pollInterval); // Cleanup on unmount
    };
  }, [jobId]);

  return <div>Status: {status}</div>;
}

// BEST: Stop polling when job completes
function JobStatusPolling({ jobId }) {
  const [status, setStatus] = useState('pending');

  useEffect(() => {
    let mounted = true;

    const pollInterval = setInterval(async () => {
      const result = await fetch(`/api/job-status/${jobId}`);
      const data = await result.json();
      if (mounted) {
        setStatus(data);
        if (data === 'completed' || data === 'failed') {
          clearInterval(pollInterval); // Stop when done
        }
      }
    }, 2000);

    return () => {
      mounted = false;
      clearInterval(pollInterval);
    };
  }, [jobId]);

  return <div>Status: {status}</div>;
}
```

**Detection:**
- Open Chrome DevTools → Performance → Record memory snapshot
- Trigger error boundary → recover → repeat
- Check if memory keeps growing (detached intervals)
- Monitor network tab: polling requests should stop after job completion

**Phase mapping:** Address in Phase 3 (frontend error boundaries + polling fixes)

**Sources:**
- [Dealing with Memory Leaks in React Error Boundaries](https://colinchjs.github.io/2023-10-06/10-14-32-750707-dealing-with-memory-leaks-in-error-boundaries-in-react/)
- [Memory Leaks in React & Next.js](https://medium.com/@essaadani.yo/memory-leaks-in-react-next-js-what-nobody-tells-you-91c72b53d84d)
- [CVE-2026-23864: React Infinite Loop DoS](https://medium.com/@teamisyncevolution/cve-2026-23864-fixing-the-react-server-component-infinite-loop-dos-flaw-fae5c37f412d)

---

## Moderate Pitfalls

Mistakes that cause operational issues, degraded UX, or moderate security concerns.

### Pitfall 7: Rate Limiting with In-Memory Store Across Multiple Server Instances

**What goes wrong:** Adding rate limiting with default in-memory store, then deploying multiple Express server instances (Cloud Run auto-scaling, Kubernetes replicas). Each instance has its own counter, so actual limit = configured_limit × num_instances.

**Why it happens:** express-rate-limit defaults to memory store for simplicity. Works in development (single instance) but fails in production (multiple instances).

**Consequences:**
- Rate limit is 10x higher than intended with 10 server instances
- Attacker can bypass limit by hitting different instances
- In ML apps: users can upload 10 files/min instead of intended 1 file/min

**Prevention:**
```javascript
// WRONG: Default memory store (only works for single instance)
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 1
});

// RIGHT: Use external store for distributed systems
const RedisStore = require('rate-limit-redis');
const redis = require('redis');
const client = redis.createClient({ url: process.env.REDIS_URL });

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 1,
  store: new RedisStore({
    client: client,
    prefix: 'rate-limit:'
  })
});
```

**Detection:**
- Deploy 2+ server instances
- Make requests to different instances (check via response headers/logs)
- Count requests before rate limit triggers
- Should be max × 1, not max × num_instances

**Phase mapping:** Address in Phase 1 (rate limiting implementation) if deploying multiple instances

**Sources:**
- [Rate Limit Bypass Techniques](https://medium.com/@rajatpatel08e/rate-limit-bypass-techniques-real-world-examples-and-how-to-defend-against-it-5fd0d82673db)
- [API Rate Limiting Best Practices](https://100plus.tools/guides/api-rate-limiting-basics)

---

### Pitfall 8: Multer File Upload Without Magic Byte Validation

**What goes wrong:** Adding file type validation checking only `mimetype` or file extension from multer. Attacker uploads malicious file (`.exe`, `.html` with XSS) by changing extension to `.csv` and setting `Content-Type: text/csv`. Multer accepts it.

**Why it happens:** `mimetype` comes from client's `Content-Type` header (trivially spoofed). File extension is also client-controlled. Both are insufficient for validation.

**Consequences:**
- Malicious executables stored in GCS
- HTML/SVG files with embedded JavaScript served from your domain (XSS)
- In ML apps: attacker bypasses CSV-only restriction to upload anything

**Prevention:**
```javascript
// WRONG: Trust client-provided mimetype
const upload = multer({
  storage: multer.memoryStorage(),
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'text/csv') {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files allowed'));
    }
  }
});

// RIGHT: Check magic bytes (file signature)
const fileType = require('file-type');

const upload = multer({
  storage: multer.memoryStorage(),
  fileFilter: async (req, file, cb) => {
    // Multer's fileFilter is synchronous, so validate in route handler instead
    cb(null, true);
  }
});

app.post('/upload', upload.single('file'), async (req, res) => {
  const buffer = req.file.buffer;

  // Check magic bytes
  const type = await fileType.fromBuffer(buffer);

  // CSV files are plain text, so file-type returns undefined
  // For CSV: check for absence of binary signatures AND validate content
  if (type) {
    return res.status(400).json({ error: 'Binary files not allowed' });
  }

  // Additional CSV validation: check first few bytes are ASCII/UTF-8
  const header = buffer.slice(0, 1024).toString('utf-8');
  if (!/^[\x20-\x7E\r\n\t]+$/.test(header)) {
    return res.status(400).json({ error: 'Invalid CSV encoding' });
  }

  // Proceed with processing
});
```

**Detection:**
- Upload `.exe` file renamed to `.csv` with `Content-Type: text/csv`
- Check if server accepts it
- Check GCS bucket for unexpected file types

**Phase mapping:** Address in Phase 1 (file upload validation)

**Sources:**
- [Secure File Upload with Magic Byte Validation](https://dev.to/myougatheaxo/secure-file-upload-with-claude-code-s3-pre-signed-urls-and-magic-byte-validation-119p)
- [How to Not Get Hacked Through File Uploads](https://www.eliranturgeman.com/2026/03/14/uploads-attack-surface/)
- [Secure File Upload with Express and Multer](https://medium.com/@tahaharbouch1/toward-secure-code-how-to-secure-file-upload-on-expressjs-using-multer-6598b1715bb4)

---

### Pitfall 9: CSV Injection in Exported Results

**What goes wrong:** Adding CSV export feature for outlier scores without sanitizing cell values. User uploads CSV with malicious formula in column name or value (`=cmd|'/c calc'!A1`). Autoencoder processes it, exports results with formula intact. When researcher opens exported CSV in Excel, formula executes.

**Why it happens:** CSV parsers/writers don't sanitize formulas by default. Developer assumes CSV is "just data" and doesn't realize spreadsheet apps execute formulas.

**Consequences:**
- Code execution on researcher's machine when opening exported CSV
- Exfiltration of researcher's files via `=WEBSERVICE()` formulas
- In ML apps: attacker uploads data, gets results, researcher downloads malicious CSV

**Prevention:**
```javascript
// WRONG: Export raw values
function exportCSV(results) {
  const rows = results.map(r => [r.id, r.score, r.flagged]);
  return rows.map(r => r.join(',')).join('\n');
}

// RIGHT: Sanitize formula characters
function sanitizeCSVCell(value) {
  const str = String(value);

  // Check if starts with formula characters
  const dangerousChars = ['=', '+', '-', '@', '\t', '\r'];
  if (dangerousChars.some(char => str.startsWith(char))) {
    return "'" + str; // Prefix with single quote to escape
  }

  // Check for dangerous patterns
  if (/cmd|powershell|DDE|WEBSERVICE/i.test(str)) {
    return "'" + str;
  }

  return str;
}

function exportCSV(results) {
  const rows = results.map(r => [
    sanitizeCSVCell(r.id),
    sanitizeCSVCell(r.score),
    sanitizeCSVCell(r.flagged)
  ]);
  return rows.map(r => r.join(',')).join('\n');
}

// BEST: Use library with built-in protection
const stringify = require('csv-stringify/sync');

function exportCSV(results) {
  return stringify(results, {
    header: true,
    escape_formulas: true // Escapes =, +, -, @, \t, \r
  });
}
```

**Detection:**
- Upload CSV with cell value `=1+1`
- Process and export results
- Open exported CSV in Excel
- Check if formula executes (shows `2`) or displays as literal (`=1+1`)

**Phase mapping:** Address in Phase 4 (CSV export feature)

**Sources:**
- [CSV Injection Prevention Best Practices](https://www.cyberchief.ai/2024/09/csv-formula-injection-attacks.html)
- [CSV Stringify escape_formulas Option](https://csv.js.org/stringify/options/escape_formulas/)
- [CSV Injection Comprehensive Guide](https://xcloud.host/csv-injection-and-how-to-prevent-it/)

---

### Pitfall 10: GCS Signed URLs with Expiration > 7 Days

**What goes wrong:** Adding signed URL generation for download links with long expiration (30 days, 90 days) for user convenience. Works fine for first 7 days, then suddenly all download links break with 403 errors.

**Why it happens:** GCS V4 signing enforces maximum 7-day (604800 second) expiration. Any URL with expiration > 7 days appears to generate successfully but breaks after 7 days. V2 signing allowed longer expiration but is less secure.

**Consequences:**
- Download links break unexpectedly after 7 days
- Users can't access their historical results
- Support tickets: "My download link doesn't work anymore"
- Recomputing signed URLs for old jobs requires access to original GCS paths

**Prevention:**
```javascript
// WRONG: Long expiration
const [url] = await bucket.file(filename).getSignedUrl({
  version: 'v4',
  action: 'read',
  expires: Date.now() + 30 * 24 * 60 * 60 * 1000 // 30 days - BREAKS AFTER 7!
});

// RIGHT: Respect 7-day limit or generate on-demand
const [url] = await bucket.file(filename).getSignedUrl({
  version: 'v4',
  action: 'read',
  expires: Date.now() + 7 * 24 * 60 * 60 * 1000 // 7 days max
});

// BEST: Generate signed URLs on-demand when user requests download
app.get('/api/download/:jobId', async (req, res) => {
  const job = await getJob(req.params.jobId);

  // Generate fresh URL with 1-hour expiration
  const [url] = await bucket.file(job.resultFile).getSignedUrl({
    version: 'v4',
    action: 'read',
    expires: Date.now() + 60 * 60 * 1000 // 1 hour
  });

  res.json({ downloadUrl: url });
});
```

**Detection:**
- Generate signed URL with 14-day expiration
- Wait 8 days (or mock system time)
- Try to access URL → should fail with 403

**Phase mapping:** Address in Phase 4 (result download feature)

**Sources:**
- [GCS Signed URLs Documentation](https://docs.cloud.google.com/storage/docs/access-control/signed-urls)
- [Managing Signed URL Risks](https://medium.com/google-cloud/managing-signed-url-risks-in-google-cloud-4d256bd58735)
- [Signed URL Expiration Discussion](https://github.com/jschneier/django-storages/issues/1021)

---

### Pitfall 11: React Polling with Stale Closure on Toast Function

**What goes wrong:** Adding job status polling with `useEffect` but including `toast` function in dependency array. Toast function reference changes on every render, causing `useEffect` to restart polling interval constantly, creating hundreds of intervals.

**Why it happens:** ESLint exhaustive-deps rule requires including all dependencies. Developer includes `toast` to satisfy linter, not realizing it causes infinite recreation of polling logic.

**Consequences:**
- Hundreds of simultaneous polling intervals
- API request spam (100s of req/sec instead of 1 req/2sec)
- Memory leak from accumulating intervals
- Rate limiting triggers, blocking legitimate requests

**Prevention:**
```javascript
// WRONG: Including toast in dependency array
function JobStatus({ jobId }) {
  const { toast } = useToast();

  useEffect(() => {
    const poll = setInterval(async () => {
      const status = await fetchStatus(jobId);
      if (status === 'failed') {
        toast({ title: 'Job failed' }); // Using toast
      }
    }, 2000);

    return () => clearInterval(poll);
  }, [jobId, toast]); // toast changes every render!
}

// RIGHT: Use ref for stable reference
function JobStatus({ jobId }) {
  const { toast } = useToast();
  const toastRef = useRef(toast);

  useEffect(() => {
    toastRef.current = toast; // Keep ref updated
  }, [toast]);

  useEffect(() => {
    const poll = setInterval(async () => {
      const status = await fetchStatus(jobId);
      if (status === 'failed') {
        toastRef.current({ title: 'Job failed' });
      }
    }, 2000);

    return () => clearInterval(poll);
  }, [jobId]); // toast not in deps
}

// BEST: Use useCallback for stable function reference
function JobStatus({ jobId }) {
  const { toast } = useToast();

  const showError = useCallback((message) => {
    toast({ title: message });
  }, []); // Empty deps = stable reference

  useEffect(() => {
    const poll = setInterval(async () => {
      const status = await fetchStatus(jobId);
      if (status === 'failed') {
        showError('Job failed');
      }
    }, 2000);

    return () => clearInterval(poll);
  }, [jobId, showError]); // showError is stable
}
```

**Detection:**
- Open browser DevTools → Network tab
- Start job polling
- Check request rate: should be 1 req/2sec, not 10+ req/sec
- Chrome DevTools → Performance → check for interval accumulation

**Phase mapping:** Address in Phase 3 (frontend polling fixes)

**Sources:**
- [Fix Stale Closure Issues in React Hooks](https://oneuptime.com/blog/post/2026-01-24-fix-stale-closure-issues-react-hooks/view)
- [Fix useEffect Dependencies Warnings](https://oneuptime.com/blog/post/2026-01-24-fix-useeffect-dependencies-warnings/view)
- [Common Stale Closure Bugs in React](https://dev.to/cathylai/common-stale-closure-bugs-in-react-57l6)

---

### Pitfall 12: Firestore Job Status Update Race Condition

**What goes wrong:** Adding progress tracking with worker updating Firestore job status throughout processing (`uploading → training → evaluating → completed`). Two concurrent worker instances process same message (Pub/Sub duplicate delivery), both update status, final status is indeterminate.

**Why it happens:** Firestore write operations without transactions are not atomic across multiple updates. Two workers reading job status, modifying, and writing back creates classic read-modify-write race condition.

**Consequences:**
- Job status flips between `completed` and `training` randomly
- Frontend shows wrong status to user
- Progress percentage jumps backward (85% → 20% → 100%)
- User cancels "training" job that actually completed

**Prevention:**
```javascript
// WRONG: Direct update without transaction
async function updateJobStatus(jobId, status) {
  await db.collection('jobs').doc(jobId).update({
    status: status,
    updatedAt: Date.now()
  });
}

// RIGHT: Use transaction for atomic read-modify-write
async function updateJobStatus(jobId, status) {
  const jobRef = db.collection('jobs').doc(jobId);

  await db.runTransaction(async (transaction) => {
    const jobDoc = await transaction.get(jobRef);

    if (!jobDoc.exists) {
      throw new Error('Job not found');
    }

    // Validate status transition (prevent backward moves)
    const currentStatus = jobDoc.data().status;
    const validTransitions = {
      'pending': ['uploading', 'failed'],
      'uploading': ['training', 'failed'],
      'training': ['evaluating', 'failed'],
      'evaluating': ['completed', 'failed']
    };

    if (validTransitions[currentStatus]?.includes(status)) {
      transaction.update(jobRef, {
        status: status,
        updatedAt: Date.now()
      });
    }
  });
}

// BEST: Make processing idempotent (prevent duplicates)
async function updateJobStatus(jobId, status, messageId) {
  const jobRef = db.collection('jobs').doc(jobId);

  await db.runTransaction(async (transaction) => {
    const jobDoc = await transaction.get(jobRef);
    const data = jobDoc.data();

    // Check if this message already processed
    if (data.processedMessages?.includes(messageId)) {
      return; // Already handled this message
    }

    transaction.update(jobRef, {
      status: status,
      updatedAt: Date.now(),
      processedMessages: [...(data.processedMessages || []), messageId]
    });
  });
}
```

**Detection:**
- Trigger Pub/Sub duplicate delivery (nack message manually)
- Watch Firestore updates in real-time console
- Check if status updates are sequential or jump around
- Monitor for status "downgrades" (completed → training)

**Phase mapping:** Address in Phase 2 (progress tracking implementation)

**Sources:**
- [Race Conditions in Firestore](https://medium.com/quintoandar-tech-blog/race-conditions-in-firestore-how-to-solve-it-5d6ff9e69ba7)
- [Firestore Transactions](https://firebase.google.com/docs/firestore/manage-data/transactions)
- [Database Concurrency and Firestore Transactions](https://medium.com/@shivamsharmaskp94/database-concurrency-and-firestore-transactions-how-mvcc-works-and-why-you-shouldnt-use-834da0d88e18)

---

## Minor Pitfalls

Mistakes that cause annoyance, confusion, or minor security concerns.

### Pitfall 13: TypeScript Strict Mode Migration with Liberal Type Assertions

**What goes wrong:** Enabling strict mode to catch type errors, but fixing 100s of errors by adding `as any` or `as unknown as Type` assertions everywhere. TypeScript compiles but provides no runtime safety—just shifted the problems under the rug.

**Why it happens:** Strict mode reveals 200+ real type issues. Developer is under time pressure, so "fixes" them with type assertions to silence compiler without actually fixing underlying issues.

**Consequences:**
- False sense of security ("TypeScript is enabled!")
- Runtime errors from type mismatches slip through
- In ML apps: `result.outlierScores as number[]` hides that API sometimes returns strings
- Future refactoring breaks because types don't reflect reality

**Prevention:**
```javascript
// WRONG: Type assertion to silence compiler
function processResults(data: any) {
  const scores = data.scores as number[];
  const mean = scores.reduce((a, b) => a + b) / scores.length; // Runtime error if strings!
}

// RIGHT: Validate at runtime
function processResults(data: unknown) {
  if (!isValidResults(data)) {
    throw new Error('Invalid results format');
  }
  // Now TypeScript knows data.scores is number[]
  const mean = data.scores.reduce((a, b) => a + b) / data.scores.length;
}

function isValidResults(data: unknown): data is { scores: number[] } {
  return (
    typeof data === 'object' &&
    data !== null &&
    'scores' in data &&
    Array.isArray(data.scores) &&
    data.scores.every(s => typeof s === 'number')
  );
}
```

**Strict migration approach:**
1. Enable strict mode in new files only (gradual migration)
2. Fix errors properly, not with `as any`
3. Add `// SAFETY: reason` comment if type assertion truly necessary
4. Track type assertions: `grep -r "as any" src/ | wc -l` should decrease, not increase

**Detection:**
- Count type assertions: `grep -r " as " src/`
- If count > 50, likely misusing assertions
- Code review: require `// SAFETY:` comment for every assertion
- Runtime monitoring: log type mismatches that assertions hid

**Phase mapping:** Address in Phase 3 (TypeScript strict mode)

**Sources:**
- [TypeScript Strict Mode Guide](https://oneuptime.com/blog/post/2026-02-20-typescript-strict-mode-guide/view)
- [How We Migrated 200K Lines to Strict TypeScript](https://dev.to/alexrogovjs/how-we-migrated-200k-lines-from-js-to-strict-typescript-3odd)
- [TypeScript Strict Settings Guide](https://develop.sentry.dev/frontend/strict-typescript-settings-guide/)

---

### Pitfall 14: Environment Variable Validation Only When Used

**What goes wrong:** Adding environment variable for GCP project ID (`process.env.GCP_PROJECT_ID`) but only checking if it's defined when making GCP API call. Server starts successfully, accepts uploads, then crashes when trying to create Vertex AI job because var is undefined.

**Why it happens:** Developer assumes env vars are set, doesn't validate at startup. Works in dev (env vars present) but fails in production (forgot to set in deployment config).

**Consequences:**
- Server starts but is non-functional
- Users can upload files but jobs fail mysteriously
- No clear error message about missing config
- In ML apps: files uploaded to GCS, Pub/Sub message sent, then worker crashes

**Prevention:**
```javascript
// WRONG: Check when using
async function createVertexAIJob(dataPath) {
  const projectId = process.env.GCP_PROJECT_ID; // Might be undefined!
  const job = await vertexai.projects.jobs.create({
    parent: `projects/${projectId}/locations/us-central1` // Crash here
  });
}

// RIGHT: Validate at startup
const requiredEnvVars = [
  'GCP_PROJECT_ID',
  'GCP_BUCKET_NAME',
  'PUBSUB_TOPIC',
  'FIRESTORE_COLLECTION',
  'NODE_ENV'
];

function validateEnv() {
  const missing = requiredEnvVars.filter(key => !process.env[key]);

  if (missing.length > 0) {
    console.error('Missing required environment variables:', missing);
    process.exit(1);
  }

  console.log('Environment validation passed');
}

// Call before starting server
validateEnv();
app.listen(8080);

// BEST: Use validation library
const envalid = require('envalid');

const env = envalid.cleanEnv(process.env, {
  GCP_PROJECT_ID: envalid.str(),
  GCP_BUCKET_NAME: envalid.str(),
  PUBSUB_TOPIC: envalid.str(),
  PORT: envalid.port({ default: 8080 }),
  NODE_ENV: envalid.str({ choices: ['development', 'production', 'test'] })
});

// Now env.GCP_PROJECT_ID is guaranteed to exist and be a string
```

**Detection:**
- Start server without required env vars
- Should exit immediately with clear error
- Should NOT start and accept requests

**Phase mapping:** Address in Phase 1 (before any other changes)

**Sources:**
- [Node.js Production Environment Variables](https://oneuptime.com/blog/post/2026-01-06-nodejs-production-environment-variables/view)
- [Validating Environment Variables in Node.js](https://medium.com/@davidminaya04/validating-environment-variables-in-node-js-c1c917a45d66)
- [Environment Variable Management Best Practices](https://www.envsentinel.dev/blog/environment-variable-management-tips-best-practices)

---

### Pitfall 15: Job Cancellation Without GCS Cleanup

**What goes wrong:** Adding job cancellation feature that updates Firestore status to `cancelled` but doesn't clean up uploaded file in GCS or kill in-progress Vertex AI job. GCS accumulates orphaned files, Vertex AI quota consumed by jobs user cancelled.

**Why it happens:** Developer implements cancellation as simple status update, doesn't consider distributed resources (GCS, Vertex AI, Pub/Sub messages in flight).

**Consequences:**
- GCS storage costs grow from orphaned files
- Vertex AI jobs run to completion despite cancellation, wasting quota
- User confused why cancelled job still shows progress
- Next upload might fail (quota exhausted from "cancelled" job)

**Prevention:**
```javascript
// WRONG: Only update status
app.post('/api/cancel/:jobId', async (req, res) => {
  await db.collection('jobs').doc(req.params.jobId).update({
    status: 'cancelled'
  });
  res.json({ success: true });
});

// RIGHT: Comprehensive cleanup
app.post('/api/cancel/:jobId', async (req, res) => {
  const jobId = req.params.jobId;
  const job = await db.collection('jobs').doc(jobId).get();
  const data = job.data();

  // 1. Update status first (prevent new operations)
  await db.collection('jobs').doc(jobId).update({
    status: 'cancelled',
    cancelledAt: Date.now()
  });

  // 2. Cancel Vertex AI job if running
  if (data.vertexAIJobId) {
    try {
      await vertexai.projects.jobs.cancel({
        name: data.vertexAIJobId
      });
    } catch (err) {
      console.error('Failed to cancel Vertex AI job:', err);
    }
  }

  // 3. Delete uploaded file from GCS
  if (data.uploadedFile) {
    try {
      await bucket.file(data.uploadedFile).delete();
    } catch (err) {
      console.error('Failed to delete uploaded file:', err);
    }
  }

  // 4. Delete intermediate results
  if (data.resultFile) {
    try {
      await bucket.file(data.resultFile).delete();
    } catch (err) {
      console.error('Failed to delete result file:', err);
    }
  }

  res.json({ success: true });
});
```

**Lifecycle management approach:**
```javascript
// Schedule cleanup for old cancelled/failed jobs
const cron = require('node-cron');

cron.schedule('0 3 * * *', async () => { // 3 AM daily
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000; // 7 days ago

  const oldJobs = await db.collection('jobs')
    .where('status', 'in', ['cancelled', 'failed'])
    .where('updatedAt', '<', cutoff)
    .get();

  for (const doc of oldJobs.docs) {
    const data = doc.data();

    // Clean up GCS files
    if (data.uploadedFile) {
      await bucket.file(data.uploadedFile).delete().catch(() => {});
    }
    if (data.resultFile) {
      await bucket.file(data.resultFile).delete().catch(() => {});
    }

    // Delete job record
    await doc.ref.delete();
  }

  console.log(`Cleaned up ${oldJobs.size} old jobs`);
});
```

**Detection:**
- Cancel job mid-processing
- Check GCS bucket: file should be deleted
- Check Vertex AI: job should show cancelled status
- Check Firestore: status should be `cancelled`

**Phase mapping:** Address in Phase 4 (job cancellation feature)

**Sources:**
- [Cleanup and Resource Management](https://app.studyraid.com/en/read/12494/404051/cleanup-and-resource-management)
- [GCS Object Lifecycle Management](https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-object-lifecycle-management-rules-in-google-cloud-storage/view)

---

### Pitfall 16: Error Response Includes Stack Trace in Production

**What goes wrong:** Adding comprehensive error handling middleware but forgetting to set `NODE_ENV=production`. Express returns full stack traces in 500 errors, leaking file paths, library versions, internal structure.

**Why it happens:** Works fine in development (want to see stack traces). Deploy to production, forget to set NODE_ENV, or deploy script doesn't set it.

**Consequences:**
- Stack traces reveal code structure to attackers
- Library versions exposed (known vulnerabilities targetable)
- File paths leak internal directory structure
- In ML apps: stack trace might reveal dataset paths, model locations

**Prevention:**
```javascript
// WRONG: Generic error handler without environment check
app.use((err, req, res, next) => {
  res.status(500).json({
    error: err.message,
    stack: err.stack // Always includes stack!
  });
});

// RIGHT: Environment-aware error handler
app.use((err, req, res, next) => {
  console.error(err.stack); // Log internally

  if (process.env.NODE_ENV === 'production') {
    // Production: minimal info
    res.status(500).json({
      error: 'Internal server error',
      requestId: req.id // For support
    });
  } else {
    // Development: full details
    res.status(500).json({
      error: err.message,
      stack: err.stack
    });
  }
});

// BEST: Use error handling library
const { errorHandler } = require('express-error-handler');

app.use(errorHandler({
  serializer: (err) => {
    if (process.env.NODE_ENV === 'production') {
      return {
        message: 'An error occurred',
        statusCode: err.statusCode || 500
      };
    }
    return {
      message: err.message,
      stack: err.stack,
      statusCode: err.statusCode || 500
    };
  }
}));
```

**Startup validation:**
```javascript
if (process.env.NODE_ENV === 'production') {
  console.log('Running in PRODUCTION mode');

  // Verify production-safe settings
  if (app.get('env') !== 'production') {
    console.warn('Warning: NODE_ENV=production but Express env is not production');
  }
} else {
  console.warn('WARNING: Not running in production mode!');
}
```

**Detection:**
- Make request that triggers 500 error
- Check response body for `stack` field
- If present in production, NODE_ENV not set correctly

**Phase mapping:** Address in Phase 1 (error handling setup)

**Sources:**
- [Express Error Handling](https://oneuptime.com/blog/post/2026-01-26-express-error-handling/view)
- [Information Leakage via Error Messages](https://cqr.company/web-vulnerabilities/information-leakage-via-error-messages/)
- [Express Error Handling Security](https://expressjs.com/en/guide/error-handling.html)

---

## Phase-Specific Warnings

Mapping pitfalls to implementation phases for this project:

| Phase Topic | Critical Pitfalls | Moderate Pitfalls | Minor Pitfalls |
|-------------|-------------------|-------------------|----------------|
| **Phase 1: Security Hardening** | 1. Session regeneration<br>2. Rate limit IPv6 bypass (CVE-2026-30827)<br>3. CORS dynamic reflection<br>4. Path traversal encoding bypass | 7. Rate limit memory store (multi-instance)<br>8. Multer magic byte validation | 14. Env var validation timing<br>16. Stack trace leakage |
| **Phase 2: Worker Reliability** | 5. Pub/Sub ack race condition | 12. Firestore race conditions | — |
| **Phase 3: Frontend Production** | 6. Error boundary memory leak<br> | 11. Polling stale closure (toast) | 13. TypeScript strict mode assertions |
| **Phase 4: Operational Features** | — | 9. CSV injection in exports<br>10. GCS signed URL expiration | 15. Job cancellation cleanup |

**Implementation order recommendations:**

1. **Start with environment validation (Pitfall 14)** — catch missing config before anything else runs
2. **Fix security critical first (Phase 1)** — blocks public deployment until fixed
3. **Worker reliability (Phase 2)** — prevents duplicate jobs and quota waste
4. **Frontend production (Phase 3)** — enables deployment
5. **Operational polish (Phase 4)** — enhances UX

**Integration pitfalls specific to Express + React + GCP Pub/Sub stack:**

| Integration Point | Pitfall | Prevention |
|-------------------|---------|------------|
| Express → GCS upload | Path traversal via user filename | Use `path.basename()` + sanitization |
| Express → Pub/Sub | Message published before file fully uploaded | Wait for GCS upload completion before publish |
| Pub/Sub → Worker | Ack before processing complete | Ack after success, extend deadline for long jobs |
| Worker → Vertex AI | No cancellation propagation | Store Vertex AI job ID, cancel on user request |
| Worker → Firestore | Concurrent status updates | Use transactions for atomic updates |
| Firestore → React | Polling with stale closures | Use refs or useCallback for stable function refs |
| React → Express | CORS bypass via reflection | Explicit allowlist, never reflect Origin header |
| Express → Browser | CSV formulas in downloads | Sanitize with `csv-stringify` escape_formulas |

---

## Confidence Assessment

| Area | Confidence | Source Quality |
|------|------------|----------------|
| Authentication | HIGH | Official docs, 2026 security research, CVE databases |
| Rate Limiting | HIGH | Recent CVE (CVE-2026-30827), official library docs |
| File Upload Security | HIGH | Multiple 2026 security guides, recent CVEs |
| GCP Pub/Sub | HIGH | Official Google Cloud docs (updated March 2026) |
| React Hooks | HIGH | Official React docs, multiple 2026 tutorials |
| Firestore Concurrency | HIGH | Official Firebase docs (updated March 2026) |
| TypeScript Migration | MEDIUM | Community best practices, no official guide |
| CSV Injection | HIGH | Multiple security sources, npm library docs |

All pitfalls sourced from 2026 documentation, recent CVEs, or official platform documentation. Research conducted 2026-03-24.

---

## Sources

### Authentication & Session Management
- [Security Best Practices for Express Session Management](https://medium.com/@kasongokakumbiguy/here-are-security-best-practices-for-session-management-nodejs-and-express-3257c2799f46)
- [Should you use Express-session for production?](https://supertokens.com/blog/should-you-use-express-session-for-your-production-app)
- [Express Session Management](https://oneuptime.com/blog/post/2026-02-02-express-sessions-cookies/view)

### Rate Limiting
- [CVE-2026-30827: express-rate-limit IPv6 Bypass](https://advisories.gitlab.com/pkg/npm/express-rate-limit/CVE-2026-30827/)
- [How to Handle IPv6 in Rate Limiting](https://oneuptime.com/blog/post/2026-03-20-ipv6-in-rate-limiting-middleware/view)
- [Rate Limit Bypass Techniques](https://medium.com/@rajatpatel08e/rate-limit-bypass-techniques-real-world-examples-and-how-to-defend-against-it-5fd0d82673db)
- [API Rate Limiting Best Practices](https://100plus.tools/guides/api-rate-limiting-basics)

### CORS Security
- [Express.js CORS Misconfiguration](https://www.sourcery.ai/vulnerabilities/javascript-express-security-cors-misconfiguration)
- [CORS Misconfigurations & Bypass](https://book.hacktricks.xyz/pentesting-web/cors-bypass)
- [Exploiting CORS Misconfiguration](https://www.intigriti.com/researchers/blog/hacking-tools/exploiting-cors-misconfiguration-vulnerabilities)

### File Upload Security
- [CVE-2026-3089: Path Traversal](https://advisories.gitlab.com/pkg/npm/@actual-app/sync-server/CVE-2026-3089/)
- [How to Not Get Hacked Through File Uploads](https://www.eliranturgeman.com/2026/03/14/uploads-attack-surface/)
- [Secure File Upload with Magic Byte Validation](https://dev.to/myougatheaxo/secure-file-upload-with-claude-code-s3-pre-signed-urls-and-magic-byte-validation-119p)
- [Secure File Upload with Express and Multer](https://medium.com/@tahaharbouch1/toward-secure-code-how-to-secure-file-upload-on-expressjs-using-multer-6598b1715bb4)

### GCP Pub/Sub
- [Pub/Sub Exactly-Once Delivery](https://oneuptime.com/blog/post/2026-01-27-pubsub-exactly-once/view)
- [Pub/Sub Message Deduplication](https://oneuptime.com/blog/post/2026-02-17-how-to-handle-pubsub-message-deduplication-in-subscriber-applications/view)
- [Troubleshooting Pull Subscriptions](https://docs.cloud.google.com/pubsub/docs/pull-troubleshooting)

### React Error Boundaries & Memory Leaks
- [CVE-2026-23864: React Infinite Loop DoS](https://medium.com/@teamisyncevolution/cve-2026-23864-fixing-the-react-server-component-infinite-loop-dos-flaw-fae5c37f412d)
- [Dealing with Memory Leaks in React Error Boundaries](https://colinchjs.github.io/2023-10-06/10-14-32-750707-dealing-with-memory-leaks-in-error-boundaries-in-react/)
- [Memory Leaks in React & Next.js](https://medium.com/@essaadani.yo/memory-leaks-in-react-next-js-what-nobody-tells-you-91c72b53d84d)

### React Hooks & Stale Closures
- [Fix Stale Closure Issues in React Hooks](https://oneuptime.com/blog/post/2026-01-24-fix-stale-closure-issues-react-hooks/view)
- [Fix useEffect Dependencies Warnings](https://oneuptime.com/blog/post/2026-01-24-fix-useeffect-dependencies-warnings/view)
- [Common Stale Closure Bugs in React](https://dev.to/cathylai/common-stale-closure-bugs-in-react-57l6)

### Firestore Concurrency
- [Race Conditions in Firestore](https://medium.com/quintoandar-tech-blog/race-conditions-in-firestore-how-to-solve-it-5d6ff9e69ba7)
- [Firestore Transactions](https://firebase.google.com/docs/firestore/manage-data/transactions)
- [Database Concurrency and Firestore MVCC](https://medium.com/@shivamsharmaskp94/database-concurrency-and-firestore-transactions-how-mvcc-works-and-why-you-shouldnt-use-834da0d88e18)

### TypeScript Strict Mode
- [TypeScript Strict Mode Guide](https://oneuptime.com/blog/post/2026-02-20-typescript-strict-mode-guide/view)
- [How We Migrated 200K Lines to Strict TypeScript](https://dev.to/alexrogovjs/how-we-migrated-200k-lines-from-js-to-strict-typescript-3odd)
- [TypeScript Strict Settings Guide](https://develop.sentry.dev/frontend/strict-typescript-settings-guide/)

### CSV Injection
- [CSV Injection Prevention Best Practices](https://www.cyberchief.ai/2024/09/csv-formula-injection-attacks.html)
- [CSV Stringify escape_formulas Option](https://csv.js.org/stringify/options/escape_formulas/)
- [CSV Injection Comprehensive Guide](https://xcloud.host/csv-injection-and-how-to-prevent-it/)

### GCS Signed URLs
- [GCS Signed URLs Documentation](https://docs.cloud.google.com/storage/docs/access-control/signed-urls)
- [Managing Signed URL Risks](https://medium.com/google-cloud/managing-signed-url-risks-in-google-cloud-4d256bd58735)
- [How to Generate Signed URLs for GCS](https://oneuptime.com/blog/post/2026-02-17-how-to-generate-and-use-signed-urls-for-google-cloud-storage-objects/view)

### Environment Variables
- [Node.js Production Environment Variables](https://oneuptime.com/blog/post/2026-01-06-nodejs-production-environment-variables/view)
- [Validating Environment Variables in Node.js](https://medium.com/@davidminaya04/validating-environment-variables-in-node-js-c1c917a45d66)
- [Environment Variable Management Best Practices](https://www.envsentinel.dev/blog/environment-variable-management-tips-best-practices)

### Error Handling
- [Express Error Handling](https://oneuptime.com/blog/post/2026-01-26-express-error-handling/view)
- [Information Leakage via Error Messages](https://cqr.company/web-vulnerabilities/information-leakage-via-error-messages/)
- [Express Error Handling Documentation](https://expressjs.com/en/guide/error-handling.html)

### Resource Cleanup
- [Cleanup and Resource Management](https://app.studyraid.com/en/read/12494/404051/cleanup-and-resource-management)
- [GCS Object Lifecycle Management](https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-object-lifecycle-management-rules-in-google-cloud-storage/view)
