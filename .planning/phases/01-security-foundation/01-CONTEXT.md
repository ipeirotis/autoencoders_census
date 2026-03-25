# Phase 1: Security Foundation - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Add authentication, CORS restrictions, rate limiting, input validation, path traversal protection, and security headers to the Express API server. Block critical security vulnerabilities before public deployment.

</domain>

<decisions>
## Implementation Decisions

### Authentication
- Email + password authentication (classic signup/login)
- Session storage in Firestore (already using Firestore for jobs — no new infrastructure)
- Email verification sent but not required to access platform (optional verification)
- Password reset via email link required for v1

### Error Response Policy
- Generic error messages to users ("File validation failed") — no internal details exposed
- Full error details logged server-side to GCP Cloud Logging for debugging
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

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `@google-cloud/firestore`: Already initialized in server/index.ts — use for session storage
- `multer`: File upload middleware already configured (50MB limit)
- GCP clients (Storage, PubSub): Already initialized, can add auth middleware around them

### Established Patterns
- Express middleware pattern: `app.use()` for global middleware
- Route handlers return JSON responses
- Error handling via try/catch with console.log (upgrade to Cloud Logging)

### Integration Points
- `createServer()` in server/index.ts — add auth middleware before routes
- `cors()` on line 45 — replace with configured CORS
- File upload route (line 56-119) — add auth check, input validation, path sanitization
- Job status route (line 122-138) — add auth check

### Current Security Gaps (from code review)
- `cors()` with no arguments = allow all origins (line 45)
- `originalName` used directly in GCS path (line 64) = path traversal risk
- No authentication middleware
- No rate limiting
- No helmet for security headers
- Errors logged to console only

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard Express security patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-security-foundation*
*Context gathered: 2026-03-25*
