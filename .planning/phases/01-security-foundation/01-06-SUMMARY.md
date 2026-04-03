---
phase: 01-security-foundation
plan: 06
subsystem: security
tags: [security, middleware, integration-testing, authentication, validation, cors, rate-limiting]
requires: [01-01, 01-02, 01-03, 01-04, 01-05]
provides: [complete-security-stack, security-integration-tests]
affects: [frontend-server, api-routes]
tech_stack:
  added: []
  patterns: [middleware-composition, security-layering]
key_files:
  created:
    - frontend/server/__tests__/integration/security.test.ts
  modified:
    - frontend/server/routes/auth.ts
    - frontend/server/routes/jobs.ts
    - frontend/server/index.ts
decisions:
  - "Apply security middleware in order: auth → rate limiting → validation"
  - "Discard user-provided filenames entirely, use UUID-based names for storage"
  - "Hide detailed validation errors from client responses (log server-side only)"
metrics:
  duration_minutes: 1
  tasks_completed: 5
  files_modified: 4
  commits: 4
  tests_added: 1
  completed_date: "2026-04-02"
---

# Phase 01 Plan 06: Wire Security Stack Summary

**One-liner:** Complete Express security middleware integration with auth, CORS, rate limiting, validation, and end-to-end integration tests verifying all security behaviors.

## What Was Built

Wired all previously created security middleware (Plans 01-05) into the Express application routes, creating a complete security stack with proper middleware ordering and comprehensive integration tests.

**Key Components:**
- Auth routes protected with input validation middleware
- Jobs routes secured with full middleware stack (auth → rate limiting → validation)
- Legacy upload route updated with auth, rate limiting, and file validation
- End-to-end integration tests covering all security behaviors
- Proper error handling and security header verification

## Tasks Completed

### Task 1: Update auth routes with validation middleware
**Commit:** 03aff8c
**Status:** Complete
**Files:** `frontend/server/routes/auth.ts`

Added `validateSignup` and `validateLogin` middleware to auth routes, ensuring input validation runs before authentication handlers.

**Key changes:**
- Imported validation chains from `middleware/validation`
- Applied validation middleware to `/signup` and `/login` routes
- Validation rejects malformed requests before processing

### Task 2: Update jobs routes with full security stack
**Commit:** dcfeec7
**Status:** Complete
**Files:** `frontend/server/routes/jobs.ts`

Wired complete security stack into jobs routes with proper middleware ordering.

**Key changes:**
- Imported all security middleware (auth, rate limiting, validation)
- Applied middleware in correct order: `requireAuth → rateLimiter → validation → handler`
- Updated `/upload-url` to use `generateSafeFilename()` for UUID-based filenames
- Updated `/start-job` with auth + rate limiting + validation
- Updated `/job-status/:id` with auth + rate limiting + validation
- Rate limiters use `req.user.id` for per-user tracking

### Task 3: Update legacy upload route with security
**Commit:** 767a04b
**Status:** Complete
**Files:** `frontend/server/index.ts`

Secured legacy direct upload route with complete security stack.

**Key changes:**
- Added `requireAuth` and `uploadLimiter` to `/api/upload` route
- Integrated `validateCSVContent()` to reject non-CSV files
- Added `generateSafeFilename()` to prevent path traversal
- Added error handler as last middleware
- Sanitized error responses (hide validation details from client)

### Task 4: Create end-to-end security integration tests
**Commit:** 8c410c8
**Status:** Complete
**Files:** `frontend/server/__tests__/integration/security.test.ts`

Created comprehensive integration test suite covering all security behaviors.

**Test coverage:**
- Authentication: Rejects unauthenticated requests, accepts authenticated requests
- Rate limiting: Blocks 6th upload within 15-minute window
- Input validation: Rejects non-UUID job IDs, validates request bodies
- CORS: Blocks unauthorized origins, allows whitelisted origins
- Security headers: Verifies X-Frame-Options, X-Content-Type-Options, etc.
- Error handling: Hides stack traces in production mode

### Task 5: Human verification checkpoint
**Status:** Approved
**Verification method:** Manual curl tests + automated test suite

**Verification results:**
- ✅ CORS blocks unauthorized origins (no Access-Control header for evil.com)
- ✅ CORS allows authorized origin (http://localhost:5173)
- ✅ Authentication required (401 Unauthorized for unauthenticated requests)
- ✅ Security headers present (X-Frame-Options: DENY, X-Content-Type-Options: nosniff)
- ✅ Input validation rejects invalid data with clear error messages
- ✅ All automated tests passing

## Deviations from Plan

None - plan executed exactly as written.

## Technical Decisions

### 1. Middleware Ordering Pattern
Applied security middleware in strict order: authentication → rate limiting → validation → handler.

**Rationale:**
- Auth first to identify user before rate limiting
- Rate limiting uses `req.user.id` for per-user tracking
- Validation runs after auth/rate limiting to fail fast
- Handler only executes if all security checks pass

### 2. UUID Filename Strategy
Completely discard user-provided filenames, using only UUID v4 for storage paths.

**Rationale:**
- Eliminates path traversal vulnerabilities
- Prevents filename collisions
- No user-controlled input in file paths
- Original filename preserved in metadata only (for UI display)

### 3. Error Message Sanitization
Hide detailed validation errors from client responses, log details server-side only.

**Rationale:**
- Prevents information leakage about validation logic
- Reduces attack surface for input fuzzing
- Provides generic "Validation failed" message to client
- Server logs contain full details for debugging

## Integration Points

### Upstream Dependencies (Requires)
- **01-01:** Environment validation, logging, error handling infrastructure
- **01-02:** CORS configuration, Helmet security headers
- **01-03:** Authentication middleware (`requireAuth`), session management
- **01-04:** Rate limiting middleware (upload, poll, download limiters)
- **01-05:** Input validation chains, CSV file validation, safe filename generation

### Downstream Consumers (Provides)
- **Complete Security Stack:** All API routes protected with auth, rate limiting, and validation
- **Security Integration Tests:** Comprehensive test suite for security verification
- **Production-Ready Routes:** Auth, jobs, and upload routes hardened for deployment

### Cross-Cutting Concerns (Affects)
- **Frontend Server:** All Express routes now have security middleware
- **API Routes:** `/auth/*`, `/jobs/*`, `/api/upload` routes secured
- **Error Handling:** Centralized error handler catches security middleware errors

## Files Changed

### Created (1)
- `frontend/server/__tests__/integration/security.test.ts` - End-to-end security integration tests

### Modified (3)
- `frontend/server/routes/auth.ts` - Added validation middleware
- `frontend/server/routes/jobs.ts` - Wired full security stack (auth + rate limiting + validation)
- `frontend/server/index.ts` - Secured legacy upload route, added error handler

## Testing Results

**Test Suite:** `frontend/server/__tests__/integration/security.test.ts`

**Results:**
- All integration tests passing
- Coverage includes authentication, rate limiting, validation, CORS, headers, error handling
- Manual verification completed via curl tests

**Test categories:**
1. Authentication: Unauthenticated/authenticated request handling
2. Rate limiting: 15-minute upload window enforcement
3. Input validation: UUID validation, request body validation
4. CORS: Origin blocking/allowing behavior
5. Security headers: Helmet header verification
6. Error handling: Stack trace hiding in production

## Requirements Satisfied

This plan satisfies requirements: SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, SEC-07, SEC-08, SEC-09, SEC-10, SEC-11, SEC-13, SEC-14, SEC-15

**Security Foundation (Complete):**
- ✅ SEC-01: Authentication required for protected routes
- ✅ SEC-02: Session management with secure cookies
- ✅ SEC-03: CORS restricts cross-origin requests
- ✅ SEC-04: Helmet security headers prevent common attacks
- ✅ SEC-05: Rate limiting prevents abuse
- ✅ SEC-06: Input validation sanitizes user input
- ✅ SEC-07: File validation prevents malicious uploads
- ✅ SEC-08: Path traversal protection via UUID filenames
- ✅ SEC-09: Error responses hide sensitive details
- ✅ SEC-10: Centralized error handling
- ✅ SEC-11: Security integration tests
- ✅ SEC-13: Middleware composition pattern
- ✅ SEC-14: Production-ready error handling
- ✅ SEC-15: Complete security stack integration

## Performance Metrics

- **Duration:** 1 minute (checkpoint resume + summary creation)
- **Tasks:** 5/5 completed (4 implementation + 1 verification)
- **Commits:** 4 (one per implementation task)
- **Files modified:** 4 (3 route files + 1 test file created)
- **Tests added:** 1 integration test suite (6 test categories)

## Next Steps

**Phase 1 Complete:** All 6 security foundation plans executed successfully.

**Ready for Phase 2:** Worker Reliability & Resilience
- Plan 02-01: Pub/Sub message validation and duplicate handling
- Plan 02-02: Firestore concurrent update protection
- Plan 02-03: Job cancellation and timeout handling
- Plan 02-04: Progress tracking and status updates
- Plan 02-05: Worker error recovery and retry logic

**Validation Required:**
- Security audit of complete Phase 1 implementation
- Penetration testing of hardened endpoints
- Load testing of rate limiting under concurrent requests

## Self-Check

Verifying all claimed artifacts exist.

**Commits:**
- ✓ FOUND: 03aff8c (Task 1: auth routes validation)
- ✓ FOUND: dcfeec7 (Task 2: jobs routes security stack)
- ✓ FOUND: 767a04b (Task 3: legacy upload route security)
- ✓ FOUND: 8c410c8 (Task 4: integration tests)

**Files:**
- ✓ FOUND: frontend/server/__tests__/integration/security.test.ts
- ✓ FOUND: frontend/server/routes/auth.ts
- ✓ FOUND: frontend/server/routes/jobs.ts
- ✓ FOUND: frontend/server/index.ts

**Result:** ✅ PASSED - All commits and files verified.
