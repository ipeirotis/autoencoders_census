# Feature Landscape: Production Web Application Features

**Domain:** ML Platform Web Application (AutoEncoder Outlier Detection)
**Researched:** 2026-03-24

## Table Stakes

Features users expect from production web applications. Missing = product feels incomplete or unprofessional.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Input Validation with Clear Errors** | Users expect immediate, actionable feedback when uploads fail. Industry standard in 2026. | Medium | Multi-layered: client-side (file type, size), server-side (MIME validation, encoding detection, schema validation). Row-level error reporting with downloadable failed-rows CSV. |
| **HTTPS/TLS** | Absolute minimum for any production web app handling user data. No exceptions in 2026. | Low | Already standard in cloud deployments. GCP handles this. |
| **API Authentication** | Production APIs without auth = critical vulnerability. Users expect secure access. | Medium | Session-based for web UI (httpOnly cookies), API keys for server-to-server. OAuth/JWT if multi-device support needed. |
| **CORS Configuration** | Open CORS = major security hole. Users expect apps to protect their data. | Low | Restrict to actual frontend domain(s). Never use `*` on authenticated endpoints. |
| **Rate Limiting** | Prevents abuse, ensures fair usage. Expected on all production APIs in 2026. | Medium | Token bucket algorithm recommended. Per-endpoint limits. Return 429 with Retry-After header. |
| **File Type Validation** | Prevents malicious uploads. Table stakes for file upload functionality. | Medium | Multi-layer: extension whitelist, MIME type verification (server-side with finfo_file), magic bytes validation. Never trust client-provided Content-Type. |
| **Path Traversal Protection** | Critical security. Exploited path traversal = data breach. | Low | Generate random UUIDs for stored filenames, discard user filenames entirely. Validate/canonicalize paths. |
| **Progress Indicators (10+ second tasks)** | Jobs taking 10-15 minutes without feedback = users assume failure, abandon task. | Medium | Multi-stage indicators required: Queued → Preprocessing → Training → Scoring. Percentage + status text. Update via Firestore .onSnapshot() listeners. |
| **Error Boundaries (React)** | Production apps should never show blank screens. Graceful degradation expected. | Low | React componentDidCatch. Display fallback UI, log errors to monitoring service. |
| **Graceful Degradation** | When components fail, core functionality should remain available. | Medium | Circuit breaker pattern for external services. Degrade non-critical features under load. |
| **File Size Limits** | Prevent DoS attacks, out-of-memory errors. | Low | Client and server validation. Clear error message with limit shown. |
| **CSV Export of Results** | Users need to use outlier scores in their analysis tools. Export = obvious necessity. | Low | Simple browser download with proper Content-Disposition headers. Consider streaming for large datasets (>10k rows). |

## Differentiators

Features that set products apart. Not expected, but valued when present. Competitive advantages.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Per-Column Outlier Explanations** | Distinguishes ML platform from black-box tools. Researchers want to know WHY a row was flagged. | Medium | Display feature importance/contribution scores per record. SHAP/LIME for deep explanations. In 2026, explainability is core requirement for ML products. |
| **Job Cancellation** | Saves compute costs, improves UX. Users recognize when they uploaded wrong file. | Medium | Requires job lifecycle management (Queued/Running/Canceled states). Transaction-safe cancellation to prevent race conditions. |
| **Row-Level Validation Feedback** | Shows which specific rows/cells failed validation, allows fix-and-reupload. | High | Download failed rows CSV with error descriptions. Validates only bad rows on re-upload. Industry leaders (CSVBox) offer this. |
| **Streaming CSV Preview** | Handles large files (>100MB) without memory issues. Shows user what was parsed. | Medium | Parse first N rows for preview. Fixes memory issues with wide datasets. |
| **Real-Time Job Queue Position** | Reduces uncertainty. Users tolerate longer waits when they know their position. | Medium | "Job #3 of 7 in queue. Estimated start: 5 min." Requires queue visibility in Pub/Sub. |
| **Automatic Schema Detection** | Reduces user effort. Platform detects column types, suggests preprocessing. | High | Analyze CSV structure, infer types, apply Rule of 9 automatically. Show user what was detected. |
| **Batch Processing** | Upload multiple CSVs, process as single job. Saves time for power users. | High | Requires job hierarchy (parent job → child tasks). Complex state management. |
| **Job History & Re-run** | Access past results without re-uploading. Compare different runs. | Medium | Store job metadata in Firestore. Link to GCS results. Allow parameterized re-runs. |
| **Email Notifications** | Long jobs = users close browser. Notification brings them back when ready. | Low | Cloud Functions trigger on job completion. Send email with results link. |
| **Webhook Support** | Programmatic integration. Appeals to technical users, enables automation. | Medium | POST job results to user-provided URL. HMAC signature for security. Retry logic. |

## Anti-Features

Features to explicitly NOT build. Tempting but wrong for this context.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Rolling Your Own Auth** | Building custom auth from scratch in 2026 = reinventing the wheel poorly. High security risk. | Use established providers (Auth0, Clerk, Supabase Auth) if complex auth needed. For MVP, API keys or session-based auth sufficient. |
| **Synchronous Processing** | 10-15 minute Vertex AI jobs cannot be synchronous. Users won't keep browser open. | Already using async Pub/Sub queue architecture. Maintain this pattern. |
| **Real-Time Streaming Results** | Autoencoder training inherently batch-based. No meaningful partial results until job completes. | Provide stage-based progress (preprocessing 50%, training 20%) but not row-by-row streaming. |
| **Mobile Native App** | Web-first approach validated. Native app = 3x development cost for minimal gain. | Responsive web design. PWA if offline access needed later. |
| **In-Browser Training** | TensorFlow.js for autoencoder = slow, limited to tiny datasets, poor UX. | Keep Vertex AI / local Python worker architecture. Browser handles UI only. |
| **Multi-User Collaboration** | Adds complexity (permissions, sharing, workspaces). Not validated need for v1.0. | Single-user jobs. Defer collaboration features to v2+ if users request. |
| **Custom Data Transformations UI** | Visual ETL builder = massive scope. Researchers comfortable with preparing CSVs. | Accept clean CSV uploads. Document expected format. Provide clear validation errors. |
| **AI-Powered Auto-Fix** | "AI fixes your data quality issues" sounds good but is unpredictable, risky for research data. | Show users what's wrong (row-level errors). Let them fix it. Preserve data integrity. |
| **File Format Conversions** | Supporting Excel, JSON, Parquet = format hell. Maintenance burden. | CSV only. Provide clear "convert to CSV first" guidance in docs. |
| **Version Control for Datasets** | Git-like versioning for data = complex. Not validated user need. | Store job history with references to original uploads. Simpler, sufficient. |

## Feature Dependencies

Dependencies between features and existing platform capabilities:

```
Existing Upload Flow → Input Validation → Row-Level Error Feedback
Existing Upload Flow → File Type Validation → Path Traversal Protection
Existing Firestore Job Tracking → Progress Indicators → Job Cancellation
Existing Firestore Job Tracking → Job History & Re-run
Existing Results Display → CSV Export
Existing Results Display → Per-Column Explanations
API Authentication → Rate Limiting (authentication identifies client for rate tracking)
Progress Indicators → Email Notifications (trigger when progress reaches 100%)
```

## MVP Recommendation

### Must-Have for Production (Table Stakes Priority)

Build these first. Without them, app is not production-ready:

1. **Security Hardening (Critical)**
   - API authentication (session-based for web UI)
   - CORS restrictions to actual domain
   - Rate limiting (token bucket, per-endpoint)
   - File type validation (multi-layer)
   - Path traversal protection (UUID filenames)

2. **Input Validation & Error Handling (Critical)**
   - CSV validation (encoding, structure, size limits)
   - Clear error messages (inline validation UX patterns)
   - React error boundaries

3. **Progress & Job Control (High Priority)**
   - Progress indicators with stages (Queued → Preprocessing → Training → Scoring)
   - Job cancellation from UI
   - Graceful degradation when external services fail

4. **Results Export (High Priority)**
   - CSV export of outlier scores
   - Proper download handling (Content-Disposition headers)

### Differentiators for v1.0 (If Time Permits)

5. **Per-Column Outlier Explanations (Medium Priority)**
   - Feature contribution scores
   - Helps users understand WHY rows were flagged
   - Aligns with 2026 ML explainability standards

6. **Row-Level Validation Feedback (Medium Priority)**
   - Download failed rows with error descriptions
   - Re-upload only bad rows after fixing
   - Matches industry-leading CSV importers

### Defer to v1.1+

- Job history & re-run (nice-to-have, not blocking)
- Email notifications (users can bookmark results URL)
- Real-time queue position (extra complexity)
- Streaming CSV preview (optimization, not blocker)
- Webhook support (enterprise feature, no validated demand)
- Batch processing (power user feature)
- Automatic schema detection (already have Rule of 9)

## Complexity Assessment

### Low Complexity (1-2 days each)
- CORS configuration
- Path traversal protection (UUID filenames)
- File size limits
- Error boundaries
- HTTPS/TLS (infrastructure already handles)
- CSV export button

### Medium Complexity (3-5 days each)
- API authentication (session-based)
- Rate limiting implementation
- File type validation (multi-layer)
- Input validation with clear errors
- Progress indicators (multi-stage)
- Job cancellation
- Per-column explanations (if model already computes contribution scores)
- Graceful degradation patterns

### High Complexity (1-2 weeks each)
- Row-level validation feedback with downloadable errors
- Streaming CSV preview for memory optimization
- Batch processing
- Automatic schema detection beyond Rule of 9

## User Experience Considerations

### Feedback Timing by Wait Duration

Based on 2026 UX research:

| Wait Time | User Expectation | Implementation |
|-----------|------------------|----------------|
| < 1 second | No indicator needed | Silent processing |
| 1-3 seconds | Loading spinner | Indeterminate spinner |
| 3-10 seconds | Progress sense | Determinate progress bar |
| 10+ seconds | Detailed status | Stage-based progress + percentage + status text |
| 10-15 minutes (Vertex AI jobs) | Background task pattern | "Fire, forget, notify" - user can leave page, poll for updates, optional email notification |

**For AutoEncoder Platform:** 10-15 minute jobs require background task UX. Key patterns:
- Collapse task to background state (user not blocked from browsing results history)
- Multi-stage progress (not just spinner)
- Job cancellation available
- Clear "safe to close browser" messaging

### Error Message Best Practices (2026)

1. **Proximity:** Place error messages directly below problematic fields/sections
2. **Plain Language:** Avoid technical jargon. "Invalid file format. Expected CSV, received .xlsx" not "MIME type mismatch"
3. **Visual Indicators Beyond Color:** Icons + borders + text (accessibility)
4. **Constructive Solutions:** "File exceeds 100MB limit. Try splitting into smaller files" not just "File too large"
5. **Inline Validation:** Catch errors as user moves out of field/section (for forms)
6. **Row-Level Specificity:** "Row 47, Column 'age': Expected number, found 'N/A'" not "Invalid data in CSV"

### Validation Layers

Modern CSV upload requires defense-in-depth:

**Client-Side (Pre-Upload):**
- File type check (.csv extension)
- File size limit (prevent large uploads that will fail)
- Basic preview (show first 5 rows)

**Server-Side (Critical Security Layer):**
- MIME type verification (finfo_file, not client Content-Type)
- File signature validation (magic bytes)
- Encoding detection (UTF-8, Windows-1252, etc.)
- Delimiter detection (commas, semicolons, tabs)
- Schema validation (required columns, type checking)
- Size limits (re-validate, don't trust client)

**Processing Layer:**
- Row-by-row validation with specific error collection
- Stream large files (don't load entire dataset into memory)
- Batch database operations for performance

### Security Hardening Checklist

Based on OWASP 2026 API Security and industry best practices:

**Authentication & Authorization:**
- [ ] API authentication implemented (sessions with httpOnly cookies for web UI)
- [ ] Never rely on client-provided tokens alone
- [ ] Short token lifetimes (15-60 min) with refresh tokens
- [ ] Validate token THEN check permissions (auth ≠ authz)

**Input Validation:**
- [ ] Whitelist approach for file extensions
- [ ] Multi-layer MIME validation (server-side only)
- [ ] Path sanitization (UUID filenames, discard user input)
- [ ] Normalize and canonicalize all paths
- [ ] Job ID validation before Firestore access

**Rate Limiting:**
- [ ] Token bucket algorithm (recommended for APIs)
- [ ] Per-endpoint limits (stricter on auth endpoints)
- [ ] Rate limit headers in responses (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset)
- [ ] 429 Too Many Requests with Retry-After header
- [ ] Redis for distributed rate limiting with atomic operations
- [ ] Rate limiting BEFORE expensive operations (auth, DB, business logic)

**CORS & Headers:**
- [ ] Explicit allowed origins (never `*` on authenticated endpoints)
- [ ] Dynamic origin reflection avoided
- [ ] X-Content-Type-Options: nosniff
- [ ] Proper Content-Type headers for all responses

**File Upload Security:**
- [ ] Store uploads outside web root (not directly accessible via URL)
- [ ] Serve files through application controllers with auth checks
- [ ] Content-Disposition headers for downloads
- [ ] Malware scanning for uploaded files (antivirus API or cloud scanning)
- [ ] Separate domain for serving user content (no shared cookies/auth state)

**Monitoring & Logging:**
- [ ] Log all authentication attempts
- [ ] Log rate limit violations
- [ ] Monitor upload failures and patterns
- [ ] Track API errors with service (Sentry, LogRocket, etc.)
- [ ] Alert on unusual patterns (spike in 429s, 403s)

## Implementation Notes

### Progress Indicator Architecture

**Firestore Schema for Job Status:**
```javascript
jobs/{jobId}: {
  status: "queued" | "preprocessing" | "training" | "scoring" | "completed" | "failed" | "canceled",
  stage: "queued" | "preprocessing" | "training" | "scoring",
  progress: 0-100, // percentage within current stage
  stageStarted: timestamp,
  estimatedCompletion: timestamp,
  error: string | null,
  canceledAt: timestamp | null,
  cancelRequested: boolean
}
```

**Frontend Pattern:**
```javascript
// Real-time listener (Firestore .onSnapshot())
const unsubscribe = db.collection('jobs').doc(jobId).onSnapshot(doc => {
  const { status, stage, progress } = doc.data();
  updateProgressUI(stage, progress);
});
```

**Stage Durations (for estimation):**
- Queued: 0-5 min (variable, depends on queue length)
- Preprocessing: ~1-2 min (CSV parsing, Rule of 9)
- Training: ~8-12 min (Vertex AI)
- Scoring: ~1-2 min (inference on all rows)

### Job Cancellation Pattern

**Safe Cancellation (Avoid Race Conditions):**
```javascript
// User clicks "Cancel" → Update Firestore
await db.collection('jobs').doc(jobId).update({ cancelRequested: true });

// Worker checks before expensive operations
const job = await getJob(jobId);
if (job.cancelRequested) {
  await updateJob(jobId, { status: 'canceled', canceledAt: new Date() });
  return; // Exit early
}
// Proceed with training...
```

**Cancellation States:**
- Queued → Canceled (immediate, job never started)
- Preprocessing → Canceled (interrupt, cleanup temp files)
- Training → Cannot cancel (Vertex AI job already running, let it complete but mark as canceled)
- Completed/Failed → Cannot cancel (job already terminal state)

### CSV Export Implementation

**Simple Download (Client-Side):**
```javascript
// For small-to-medium results (<10k rows)
const csv = convertToCSV(results); // results array
const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
const link = document.createElement('a');
link.href = URL.createObjectURL(blob);
link.download = `outlier-scores-${jobId}.csv`;
link.click();
```

**Streaming Download (Server-Side for Large Results):**
```javascript
// For large results (>10k rows), stream from GCS
app.get('/api/jobs/:jobId/export', async (req, res) => {
  const { jobId } = req.params;
  const file = storage.bucket(bucket).file(`results/${jobId}.csv`);

  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="outlier-scores-${jobId}.csv"`);

  file.createReadStream()
    .on('error', err => res.status(500).send('Export failed'))
    .pipe(res);
});
```

### Per-Column Explanation Data

**If model already computes reconstruction errors per feature:**
```javascript
// Expand results schema
results: [
  {
    rowIndex: 47,
    reconstructionError: 0.82,
    isOutlier: true,
    columnContributions: [
      { column: 'age', contribution: 0.35, originalValue: 'N/A', reconstructedValue: 42 },
      { column: 'income', contribution: 0.28, originalValue: 200000, reconstructedValue: 65000 },
      { column: 'education', contribution: 0.19, originalValue: 'HS', reconstructedValue: 'College' }
    ]
  }
]
```

**UI Pattern:**
- Default view: List of outlier rows with overall scores
- Expandable row detail: Click row → Show top 5 contributing columns
- Visual: Horizontal bar chart showing contribution percentages
- Tooltip: "This value differs significantly from the model's expectation"

## Sources

### API Authentication & Security
- [API Authentication Best Practices in 2026 - DEV Community](https://dev.to/apiverve/api-authentication-best-practices-in-2026-3k4a)
- [Web Application Authentication Best Practices in 2026](https://www.devstars.com/blog/web-application-authentication/)
- [REST API Authentication: 7 Methods Compared (2026 Guide)](https://www.knowi.com/blog/4-ways-of-rest-api-authentication-methods/)
- [Session-Based Authentication vs Token-Based Authentication (2026)](https://securityboulevard.com/2026/01/session-based-authentication-vs-token-based-authentication-key-differences-explained/)
- [OWASP API Security: Top 10 Security Risks & Remedies for 2026](https://blog.axway.com/learning-center/digital-security/risk-management/owasps-api-security)

### CSV Upload & Validation
- [CSV Import with Validation and Error Handling (February 2026)](https://medium.com/@sohail_saifi/csv-import-with-validation-and-error-handling-when-users-upload-the-messiest-data-610329dc4d48)
- [Show row-level error messages in imports - CSVBox Blog](https://blog.csvbox.io/row-level-errors-csv/)
- [The Ultimate Guide to CSV File Validation (2026)](https://disbug.io/en/blog/ultimate-guide-csv-file-validation-data-quality-systems/)
- [How to Automate CSV Data Quality in Real Time (2026)](https://www.integrate.io/blog/how-to-automate-csv-data-quality-in-real-time-2025/)

### Security Hardening
- [CORS Security: Beyond Basic Configuration](https://www.aikido.dev/blog/cors-security-beyond-basic-configuration)
- [API Security in 2026: The 10 Vulnerabilities That Keep Getting Exploited](https://garnetgrid.com/insights/api-security-vulnerabilities)
- [Path Traversal Prevention - OWASP Foundation](https://owasp.org/www-community/attacks/Path_Traversal)
- [How to Not Get Hacked Through File Uploads (2026)](https://www.eliranturgeman.com/2026/03/14/uploads-attack-surface/)
- [File Upload Security - OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)

### Rate Limiting
- [API Rate Limiting: Strategies, Implementation, and Best Practices (2026)](https://aiforeverthing.com/blog/api-rate-limiting-guide.html)
- [API Rate Limiting Implementation (February 2026)](https://oneuptime.com/blog/post/2026-02-20-api-rate-limiting-strategies/view)
- [Rate Limiting AI APIs with FastAPI (February 2026)](https://dasroot.net/posts/2026/02/rate-limiting-ai-apis-async-middleware-fastapi-redis/)

### Progress Indicators & UX
- [UI patterns for async workflows, background jobs, and data pipelines](https://blog.logrocket.com/ux-design/ui-patterns-for-async-workflows-background-jobs-and-data-pipelines)
- [Progress Trackers and Indicators – With 6 Examples To Do It Right](https://userguiding.com/blog/progress-trackers-and-indicators)
- [Designing Better Loading and Progress UX](https://smart-interface-design-patterns.com/articles/designing-better-loading-progress-ux/)
- [Progress Indicator UI Design: Best practices (2026)](https://mobbin.com/glossary/progress-indicator)

### Error Handling & UX
- [Error Boundaries in React: The Safety Net Every Production App Needs (February 2026)](https://saraswathi-mac.medium.com/error-boundaries-in-react-the-safety-net-every-production-app-needs-f85809bd5563)
- [How to Implement React Error Boundaries for Resilient UIs (February 2026)](https://oneuptime.com/blog/post/2026-02-20-react-error-boundaries/view)
- [Graceful Degradation: Handling Errors Without Disrupting User Experience](https://medium.com/@satyendra.jaiswal/graceful-degradation-handling-errors-without-disrupting-user-experience-fd4947a24011)
- [Error-Message Guidelines - Nielsen Norman Group](https://www.nngroup.com/articles/error-message-guidelines/)
- [Designing User-friendly Form Error Messages](https://clearout.io/blog/form-error-messages/)

### Job Management
- [Background Job Management - Convex](https://stack.convex.dev/background-job-management)
- [Firebase 2026: The Complete Advanced Guide](https://medium.com/@ramankumawat119/firebase-2026-the-complete-advanced-guide-for-modern-app-developers-7d24891010c7)
- [Real-Time Data Handling with Firestore](https://dev.to/itselftools/real-time-data-handling-with-firestore-tracking-pending-orders-4o8l)

### ML Explainability
- [What Are Model Interpretability Techniques in AI (2026)?](https://tolumichael.com/what-are-model-interpretability-techniques-ai-2026/)
- [Improving AI models' ability to explain their predictions - MIT News (March 2026)](https://news.mit.edu/2026/improving-ai-models-ability-explain-predictions-0309)
- [What is model explainability? (2026 Guide)](https://aiopsschool.com/blog/model-explainability/)
