---
phase: 04-operational-features
plan: 06
subsystem: operational-features
tags: [file-management, ui, lifecycle, expired-jobs]
dependency_graph:
  requires: [04-03]
  provides: [expired-job-ui, manual-file-deletion]
  affects: [frontend-client, frontend-server]
tech_stack:
  added: []
  patterns: [age-based-expiration-check, AlertDialog-confirmation, best-effort-deletion]
key_files:
  created:
    - frontend/client/components/results/DeleteJobDialog.tsx
  modified:
    - frontend/server/routes/jobs.ts
    - frontend/client/pages/JobProgress.tsx
decisions:
  - decision: "Separate DELETE /jobs/:id/files endpoint from cancellation endpoint"
    rationale: "Cancellation is for running jobs, file deletion is for completed jobs. Different semantics and constraints."
    alternatives: ["Single endpoint with mode parameter", "PATCH endpoint with action field"]
  - decision: "Best-effort GCS file deletion (continue on errors)"
    rationale: "Files may already be deleted by lifecycle rules or previous manual deletion. Continue cleanup even if some files missing."
    alternatives: ["Fail-fast on first error", "Pre-check file existence"]
  - decision: "Client-side age check using isJobExpired helper"
    rationale: "Immediate UI feedback without server round-trip. Matches GCS lifecycle rule behavior (7-day retention)."
    alternatives: ["Server-side computed field", "Firestore TTL collection"]
  - decision: "Check both isJobExpired and filesExpired flags"
    rationale: "Handles both automatic expiration (GCS lifecycle) and manual deletion. filesExpired flag set by DELETE endpoint."
    alternatives: ["Single source of truth", "Computed expiration field only"]
metrics:
  duration: 300
  completed: 2026-04-07
  tasks_completed: 4
  files_created: 1
  files_modified: 2
---

# Phase 04 Plan 06: Expired Job UI and Manual File Deletion Summary

**One-liner:** Implemented expired job detection with 7-day age check, expiration message UI, download button hiding, and manual file deletion with AlertDialog confirmation for completed jobs.

## Overview

Added frontend UI for handling expired jobs (after GCS lifecycle auto-deletion) and manual file deletion feature for completed jobs. Users now see clear "Files expired" messages when accessing jobs older than 7 days and can manually delete files before expiration via delete button with confirmation dialog.

This plan completes the file lifecycle management story started in plan 04-03 (GCS lifecycle rules). After automatic deletion, users receive appropriate UI feedback instead of broken download links.

## What Was Built

### 1. Manual File Deletion Endpoint (Task 1)
**File:** `frontend/server/routes/jobs.ts`

Added `DELETE /jobs/:id/files` endpoint (separate from job cancellation endpoint):
- Only allows deletion of completed/failed/canceled jobs (not running jobs)
- Best-effort deletion of GCS uploaded file and result files
- Updates Firestore `filesExpired` flag to track manual deletion state
- Returns `filesDeleted` count for user feedback
- Error handling continues cleanup even if some files already deleted

**Key distinction:** DELETE /jobs/:id for cancellation (running jobs), DELETE /jobs/:id/files for file deletion (completed jobs).

### 2. Delete Confirmation Dialog (Task 2)
**File:** `frontend/client/components/results/DeleteJobDialog.tsx`

Created AlertDialog component for manual file deletion:
- Trash2 icon and outline variant styling (visual distinction from download button)
- Clear confirmation message explaining permanent deletion
- Toast notifications for success/error feedback
- TanStack Query cache invalidation to update filesExpired state
- Loading state during deletion

### 3. Expired Job UI Integration (Task 3)
**File:** `frontend/client/pages/JobProgress.tsx`

Enhanced JobProgress page with expired job handling:
- `isJobExpired()` helper function checks if createdAt + 7 days < now
- Yellow alert box with "Files expired - data deleted after 7-day retention period" message
- Download button hidden for expired jobs
- Manual delete button shown for non-expired completed jobs
- Checks both `isJobExpired(createdAt)` (automatic) and `job.filesExpired` (manual) flags

### 4. Worker Validation Messages Verification (Task 4)
**File:** `worker.py`

Verified that worker already provides descriptive error messages for OPS-12:
- File size: `"CSV file too large: {size}MB (max 100MB)"`
- Encoding: `"Encoding error with {encoding}: {error}"`
- Structure: `"CSV must have at least {N} rows/columns (found {X})"`
- Parsing: `"CSV parsing error: {details}"`

No changes needed - requirement OPS-12 already satisfied from Phase 2 implementation.

## Deviations from Plan

None - plan executed exactly as written.

## Technical Implementation Details

### Age-Based Expiration Check Pattern
```typescript
function isJobExpired(createdAt: Date | string): boolean {
  const created = createdAt instanceof Date ? createdAt : new Date(createdAt);
  const expirationDate = new Date(created);
  expirationDate.setDate(expirationDate.getDate() + 7);
  return new Date() > expirationDate;
}
```

Handles both Date objects and ISO string timestamps from Firestore. Matches GCS lifecycle rule behavior (7-day retention).

### Dual Expiration Flag Check
```typescript
{(isJobExpired(job.createdAt) || job.filesExpired) && (
  <div>Files expired message...</div>
)}
```

Checks both automatic expiration (age-based) and manual deletion (filesExpired flag). Ensures consistent UI regardless of deletion method.

### Best-Effort Deletion Pattern
```typescript
try {
  await storage.bucket(BUCKET_NAME).file(gcsFileName).delete();
  filesDeleted++;
} catch (error) {
  logger.warn('Failed to delete GCS file (may already be deleted)', { error });
  // Continue with other cleanup steps
}
```

Continues cleanup even if some files missing (may have been deleted by lifecycle rules or previous manual deletion). Logs warnings but doesn't fail the entire operation.

## Testing Notes

Manual testing scenarios:
1. **Expired job UI:** Mock job with createdAt 8 days ago → should see expiration message, no download button
2. **Delete dialog:** Click "Delete Files" → confirmation appears → click "Delete Files" → files deleted, toast shown
3. **GCS verification:** After deletion, check `gsutil ls gs://${BUCKET_NAME}/uploads/` → file should be gone
4. **filesExpired flag:** After deletion, job document should have `filesExpired: true`
5. **Running job protection:** Try DELETE /jobs/:id/files on running job → should return 400 error

## Requirements Satisfied

- **OPS-12:** Descriptive error messages (verified in worker.py - already complete)
- **Must-haves:** All 6 truths satisfied:
  - ✅ User sees 'Files expired' message for expired jobs
  - ✅ Download button hidden for expired jobs (age check)
  - ✅ User can manually delete job files via delete button
  - ✅ Delete confirmation dialog appears before deletion
  - ✅ Manual deletion removes GCS files but keeps Firestore metadata
  - ✅ Worker validation error messages verified as descriptive

## Performance & Metrics

- **Execution time:** 5 minutes
- **Tasks completed:** 4/4
- **Files created:** 1
- **Files modified:** 2
- **Commits:** 4 (62d91e1, c1c0c8f, 5c69c50, 4162a0a)

## Next Steps

1. Execute remaining Phase 4 plans (04-04, 04-05 if any)
2. Run full integration test suite
3. Verify expired job UI behavior with real GCS lifecycle deletion
4. Consider adding manual cleanup script for Firestore job documents (future enhancement)

## Self-Check: PASSED

**Created files exist:**
```
✓ frontend/client/components/results/DeleteJobDialog.tsx
```

**Modified files exist:**
```
✓ frontend/server/routes/jobs.ts
✓ frontend/client/pages/JobProgress.tsx
```

**Commits exist:**
```
✓ 62d91e1 - feat(04-06): add DELETE /jobs/:id/files endpoint for manual file deletion
✓ c1c0c8f - feat(04-06): create DeleteJobDialog component with AlertDialog confirmation
✓ 5c69c50 - feat(04-06): add expired job UI and manual delete button to JobProgress page
✓ 4162a0a - chore(04-06): verify worker validation error messages (OPS-12)
```

**All files and commits verified as present.**
