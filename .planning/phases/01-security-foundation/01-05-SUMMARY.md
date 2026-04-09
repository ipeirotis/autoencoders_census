---
phase: 01-security-foundation
plan: 05
subsystem: security-input-validation
tags: [security, validation, file-security, path-traversal]
dependency_graph:
  requires: [01-01, 01-03]
  provides: [input-validation, csv-validation, path-protection]
  affects: [auth-routes, jobs-routes, upload-routes]
tech_stack:
  added:
    - express-validator@7.3.1
    - file-type@21.3.4
  patterns:
    - express-validator chains
    - TDD (test-driven development)
    - file content inspection
    - UUID-based filenames
    - path normalization
key_files:
  created:
    - frontend/server/middleware/validation.ts
    - frontend/server/utils/fileValidation.ts
    - frontend/server/__tests__/middleware/validation.test.ts
    - frontend/server/__tests__/utils/fileValidation.test.ts
  modified:
    - frontend/package.json
    - frontend/package-lock.json
decisions:
  - decision: "Use express-validator for input validation"
    rationale: "Industry-standard library with built-in sanitization, clear error messages, and composable validation chains"
    alternatives: ["Manual validation", "Joi", "Yup"]
  - decision: "Use file-type library for binary file detection"
    rationale: "Detects file types by magic bytes, not extensions. Prevents binary files renamed to .csv from being processed"
    alternatives: ["Extension-only check", "Manual magic byte detection"]
  - decision: "UUID v4 for all uploaded filenames"
    rationale: "Eliminates path traversal via filename, prevents filename collisions, no user-controlled input in paths"
    alternatives: ["Sanitized user filenames", "Hash-based filenames"]
  - decision: "Separate validation middleware from route handlers"
    rationale: "Reusable validation chains, DRY principle, clear separation of concerns, easier testing"
    alternatives: ["Inline validation in routes"]
metrics:
  duration: "25m"
  tasks_completed: 5
  tests_added: 39
  files_created: 4
  files_modified: 2
  completed_date: "2026-03-30"
---

# Phase 01 Plan 05: Input Validation & File Security Summary

**One-liner:** Comprehensive input validation using express-validator chains and CSV content inspection with path traversal protection via UUID-based filenames.

## What Was Built

### Input Validation Middleware
Created express-validator chains for all user inputs:
- **Auth routes**: Email normalization, password length (8-128 chars), clear field-level error messages
- **Jobs routes**: UUID validation for job IDs, CSV filename validation, GCS filename presence check
- **Error handling**: Standardized 400 responses with `{ error: "Validation failed", details: [{ field, message }] }` format

### CSV Content Validation
Implemented multi-layer CSV validation:
1. **Binary file detection**: Uses file-type library to detect files by magic bytes (PNG, ZIP, PDF, etc.)
2. **UTF-8 encoding check**: Validates text encoding, rejects invalid sequences
3. **CSV structure validation**: Requires minimum 2 rows (header + data), verifies comma separators
4. **Windows/Unix compatibility**: Handles both CRLF and LF line endings

### Path Traversal Protection
Built two utilities to prevent path traversal attacks:
- **generateSafeFilename**: Creates UUID-based paths (`uploads/{userId}/{uuid}.csv`), never uses user-provided names
- **sanitizePath**: Normalizes paths with `path.resolve`, blocks `../` and absolute paths outside upload directory

## Tasks Completed

| Task | Name | Commit | Files | Tests |
|------|------|--------|-------|-------|
| 0 | Install validation dependencies | c63f082 | package.json, package-lock.json | - |
| 1 | Create express-validator chains for auth routes | c61f807 | validation.ts, validation.test.ts | 17 |
| 2 | Create validation chains for jobs routes | c61f807 | (same as Task 1) | (included) |
| 3 | Create CSV content validation utility | 6e5dc1e | fileValidation.ts, fileValidation.test.ts | 12 |
| 4 | Create path traversal protection utilities | 8140e47 | fileValidation.ts, fileValidation.test.ts | 10 |

**Note:** Tasks 1 and 2 were completed in a single commit since they both extend the same `validation.ts` module with related functionality.

## Verification Results

All automated tests pass:
- ✅ **17 tests** for validation middleware (auth + jobs validation)
- ✅ **12 tests** for CSV content validation (binary detection, encoding, structure)
- ✅ **10 tests** for path traversal protection (UUID generation, path sanitization)
- ✅ **117 total frontend tests** passing

### Security Properties Verified
- Invalid emails rejected with "Invalid email" message
- Passwords < 8 chars rejected
- Non-UUID job IDs rejected
- Binary files renamed to .csv rejected (PNG, ZIP, PDF)
- CSV files without commas rejected
- Path traversal attempts (`../../etc/passwd`) blocked
- UUIDs used for all file storage (no user-controlled filenames)

## Deviations from Plan

None. Plan executed exactly as written.

## Integration Points

### Ready for Plan 06 (Route Integration)
- `validateSignup`, `validateLogin` ready for auth routes
- `validateJobId` ready for jobs/:id routes
- `validateUploadUrl` ready for upload-url endpoint
- `validateStartJob` ready for start-job endpoint
- `validateCSVContent` ready for file upload handler
- `generateSafeFilename` ready for GCS upload path generation

### Dependencies Satisfied
- Requires 01-01 (environment validation) ✅
- Requires 01-03 (rate limiting) ✅
- Provides input validation for Plan 06 route integration

## Technical Decisions

### Express-Validator Pattern
Validation chains compose multiple checks with clear error messages:
```typescript
export const validateSignup = [
  body('email').isEmail().withMessage('Invalid email').normalizeEmail().trim(),
  body('password').isLength({ min: 8, max: 128 }).withMessage('Password must be 8-128 characters'),
  handleValidationErrors,
];
```

Benefits:
- Clear, declarative validation rules
- Automatic sanitization (normalizeEmail, trim)
- Field-level error details for frontend display
- Reusable across routes

### File-Type Binary Detection
Checks magic bytes before trusting file extensions:
```typescript
const type = await fileTypeFromBuffer(buffer);
if (type) {
  return { valid: false, reason: 'File appears to be binary, not CSV' };
}
```

Prevents attacks like:
- Malware.exe renamed to data.csv
- PNG/ZIP files disguised as CSV
- Executable payloads in upload directory

### UUID-Based Filenames
User-provided filenames are never used in storage:
```typescript
export function generateSafeFilename(userId: string): string {
  const fileId = uuidv4();
  return `uploads/${userId}/${fileId}.csv`;
}
```

Security benefits:
- Zero path traversal risk (no user input in path)
- No filename collisions
- Predictable path structure for cleanup
- No special character handling needed

## Testing Strategy

### TDD Approach
All tasks followed TDD cycle:
1. **RED**: Write failing tests first (define expected behavior)
2. **GREEN**: Implement minimal code to pass tests
3. **REFACTOR**: Clean up (minimal in this plan)

### Test Coverage
- **Unit tests**: Each validation chain tested in isolation
- **Integration tests**: Validation chains tested with Express routes using supertest
- **Edge cases**: Binary files, encoding issues, path traversal variations
- **Positive cases**: Valid inputs accepted without errors

## Self-Check: PASSED

### Created Files
- ✅ FOUND: frontend/server/middleware/validation.ts
- ✅ FOUND: frontend/server/utils/fileValidation.ts
- ✅ FOUND: frontend/server/__tests__/middleware/validation.test.ts
- ✅ FOUND: frontend/server/__tests__/utils/fileValidation.test.ts

### Commits
- ✅ FOUND: c63f082 (Task 0 - dependencies)
- ✅ FOUND: c61f807 (Task 1 - auth validation)
- ✅ FOUND: 6e5dc1e (Task 3 - CSV validation)
- ✅ FOUND: 8140e47 (Task 4 - path protection)

### Tests
- ✅ All 117 frontend tests passing
- ✅ All validation tests passing
- ✅ All file security tests passing

## Next Steps

Plan 06 (Route Integration) will:
- Apply validation middleware to auth routes (POST /auth/signup, POST /auth/login)
- Apply validation middleware to jobs routes (GET /jobs/:id, POST /upload-url, POST /start-job)
- Integrate CSV validation in file upload handler
- Use generateSafeFilename for GCS object paths
- Test complete request flow with validation

## Open Questions

None. All requirements satisfied, ready for route integration.
