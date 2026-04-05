# Phase 3: Frontend Production - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Frontend provides production-quality UX with error recovery, progress feedback, and job control for 10-15 minute async workflows. Includes: React error boundaries with recovery UI, multi-stage progress indicator showing Queued → Preprocessing → Training → Scoring stages, job cancellation UI, polling lifecycle cleanup, build fixes (missing dependencies, TypeScript strict mode), and memory leak prevention.

</domain>

<decisions>
## Implementation Decisions

### Error Recovery UX
- Full-page error state when components crash (not inline or toast-only)
- Reload page button as primary recovery action
- Technical details (stack trace, component name) hidden by default, expandable on click
- Component-specific error boundaries for high-risk components (Preview, Results) — not just global boundary
- Preview component error → show error in preview area only, rest of app continues working
- Results component error → show error in results area only, rest of app continues working

### Progress Indicator Design
- Dedicated progress page at /job/:id route (not modal or inline)
- User navigates to progress page after upload completes
- Step indicator with badges: (1) Queued → (2) Preprocessing → (3) Training → (4) Scoring
- Current step highlighted, completed steps show checkmarks
- Progress bar shows BOTH stage percent AND overall job percent
  - Example: "Training (65% of this stage)" + "Overall: 75% complete"
- Additional info displayed:
  - Elapsed time ("2 minutes 30 seconds" since job started)
  - Estimated time remaining ("~8 minutes left" based on average)
  - File name & size (which CSV is being processed)
  - Job ID not shown by default (can add if needed for support)

### Claude's Discretion
- Error boundary implementation details (class component vs library)
- Error boundary fallback UI styling (layout, colors, copy)
- Step indicator styling (badge colors, checkmark icons, connecting lines)
- Progress bar implementation (shadcn/ui Progress component or custom)
- Polling interval frequency (research typical React Query patterns)
- Polling cleanup implementation (useEffect dependencies, abort controllers)
- TypeScript strict mode migration strategy (incremental vs all-at-once)
- Build dependency resolution order (which missing deps to add first)
- GCP client instance deduplication approach
- Route conflict resolution strategy (duplicate job-status routes)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **shadcn/ui components** (51 components in frontend/client/components/ui/):
  - Badge: For step indicator stage labels
  - Progress: For progress bar visualization
  - Alert/AlertDialog: For error messages and confirmations
  - Card: For grouping progress info
  - Button: For reload/recovery actions
- **useToast hook**: For transient notifications (job started, cancelled, errors)
- **TanStack Query**: Already used for data fetching — can extend for polling with refetchInterval
- **Existing components**:
  - Dropzone.tsx: File upload UI
  - PreviewTable.tsx: CSV preview (high-risk component needing error boundary)
  - ResultCard.tsx: Results display (high-risk component needing error boundary)
  - NaNCheckbox.tsx: NaN handling controls

### Established Patterns
- **React 18** with functional components and hooks
- **Vite** for builds and dev server
- **Tailwind CSS** for styling (defined in tailwind.config.ts)
- **Path aliases**: @/* for client/, @shared/* for shared/
- **Prettier** formatting (2-space indentation, trailing commas)
- **TypeScript** currently with strict: false (FE-16/FE-17 requires enabling strict mode)

### Integration Points
- **Upload flow** (Dropzone.tsx): After successful upload, navigate to /job/:id progress page
- **Firestore status polling**: Worker writes status to jobs collection, frontend reads via Firestore client
- **Route structure**: Need to add /job/:id route for progress page
- **Error logging**: Errors should log to server-side (consistent with Phase 1 decision: generic messages to users, full details server-side)

### Current Frontend Gaps (from REQUIREMENTS.md)
- **Missing dependencies**: lib/utils.ts (FE-11), react-router-dom (FE-12), serverless-http (FE-13)
- **Missing npm scripts**: build:client (FE-14), dev:server (FE-15)
- **TypeScript strict mode disabled**: tsconfig.json has strict: false (FE-16), violations exist (FE-17)
- **GCP client duplication**: Multiple Storage/Firestore/PubSub instances created (FE-18)
- **Duplicate routes**: job-status route defined in both index.ts and routes/jobs.ts (FE-19)
- **Port mismatch**: Server uses different port than documented (FE-20)
- **CSV parser memory issue**: Non-streaming preview crashes on large files (FE-21)
- **File type validation gap**: Click-upload path lacks validation (only drag-drop has it) (FE-22)

</code_context>

<specifics>
## Specific Ideas

**Progress page navigation flow:**
1. User uploads CSV via Dropzone
2. Upload completes → navigate to /job/:id
3. Progress page shows step indicator + dual progress bars + elapsed/remaining time + file info
4. Poll Firestore every N seconds for status updates
5. When job completes → show results or error

**Error boundary hierarchy:**
- Root error boundary wraps entire app → catches App-level crashes, shows full-page error
- PreviewTable error boundary → catches preview crashes, shows inline error in preview area
- ResultCard error boundary → catches results display crashes, shows inline error in results area

**Technical details expansion:**
- Default view: "Something went wrong" + Reload button
- Expandable section: "Show details ▼" → reveals stack trace + component name + timestamp
- Copy to clipboard button for error details (helps users report bugs)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-frontend-production*
*Context gathered: 2026-04-05*
