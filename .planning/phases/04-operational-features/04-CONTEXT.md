# Phase 4: Operational Features - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can export outlier results as CSV files with formula injection protection, cancel running jobs with full resource cleanup (GCS files + Vertex AI jobs), see per-column outlier contribution scores in results UI to understand which survey questions were anomalous, and benefit from automatic cleanup of old files via GCS lifecycle rules. Maintainer learns GitHub PR workflow for collaborating with IliasTriant.

</domain>

<decisions>
## Implementation Decisions

### CSV Export Location & Trigger
- Download button appears on /job/:id progress page when job completes
- Not on upload page, not both locations — single consistent location
- Button enabled only when job status is "complete" (not queued/processing/error)

### CSV Export Content
- Export outlier rows + scores only (not full dataset, not summary-only)
- Columns: all original data columns from uploaded CSV + `outlier_score` column
- Smaller focused file — user gets detected outliers with context
- Non-outlier rows excluded from export

### CSV Export Security
- Prevent Excel formula injection by prefixing dangerous characters
- Dangerous chars: `=`, `+`, `-`, `@` at start of cell
- Mitigation: prepend single quote (`'`) to cells starting with dangerous chars
- Preserves data readability (e.g., `=-5` becomes `'=-5`, Excel displays as text)

### Per-Column Contribution Scores Display
- Expandable row detail pattern (click outlier row to expand inline panel)
- Similar to GitHub PR file list or Slack thread expansion
- Expanded panel shows horizontal bar chart of per-column contributions
- All columns shown, sorted by contribution (high to low)
- Not limited to top 5 or top 10 — user sees complete picture, scrollable if needed

### Per-Column Contribution Scores Format
- Horizontal bar chart using shadcn/ui Progress component
- Visual bars showing relative contribution (like GitHub commit stats)
- Each bar labeled with column name + contribution percentage
- Color-coded: high contribution (red/orange), medium (yellow), low (gray/green)
- Sorted descending by contribution value

### File Retention Policy
- 7-day retention for all GCS files (uploads and results)
- Single retention period, not different for uploads vs results
- GCS lifecycle rules automatically delete files after 7 days
- Firestore job metadata persists even after GCS files deleted

### Manual File Deletion
- Delete button on /job/:id page for manual cleanup before 7-day expiration
- Separate from job cancellation (cancellation is for running jobs, deletion is for completed jobs)
- Delete action removes: uploaded CSV from GCS, result files from GCS, keeps Firestore metadata
- Confirmation dialog before deletion (prevent accidental data loss)

### Expired Job Handling
- After GCS files auto-delete (7 days), Firestore job record remains
- User accessing /job/:id for expired job sees: job metadata (filename, upload date, status) + "Files expired - data deleted after 7 days" message
- No download button shown for expired jobs
- No automatic Firestore cleanup — job history preserved indefinitely

### Claude's Discretion
- Backend API endpoint design for CSV export (/api/jobs/:id/export vs /api/export/:id)
- CSV generation implementation (Papa Parse library, streaming approach, buffer size)
- Per-column contribution score computation algorithm (how to decompose reconstruction_loss from evaluate/outliers.py)
- Expandable row UI animation (collapse/expand transition, icon choice)
- Bar chart color palette and thresholds for high/medium/low contribution
- GCS lifecycle rule configuration syntax (Terraform, gcloud CLI, or console)
- Job cancellation GCS cleanup implementation (immediate sync delete vs background task)
- Job cancellation Vertex AI cancellation API call details
- GitHub PR workflow documentation format (README section, CONTRIBUTING.md, or separate doc)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **shadcn/ui Table component**: For displaying outlier rows with expandable detail
- **shadcn/ui Progress component**: For horizontal bar chart of per-column contributions
- **shadcn/ui Button component**: For download CSV and delete buttons
- **shadcn/ui AlertDialog component**: For delete confirmation dialog
- **shadcn/ui Collapsible component**: For expandable row detail pattern
- **shadcn/ui Badge component**: For "expired" status indicator
- **Job progress page** (frontend/client/pages/JobProgress.tsx): Existing page where download button will appear
- **useJobCancellation hook** (frontend/client/hooks/useJobCancellation.ts): DELETE /api/jobs/:id endpoint exists, currently only updates Firestore — extend to clean up GCS and Vertex AI

### Established Patterns
- **TanStack Query** for API calls (use for CSV export download trigger)
- **Firestore jobs collection** stores job metadata (status, filename, uploadDate, fileSize)
- **GCS bucket structure**: uploads/ prefix for uploaded CSVs, results/ prefix for outlier result files
- **evaluate/outliers.py**: Computes per-row reconstruction_loss — need to decompose into per-column contributions
- **ResultCard component**: Shows success/error messages, not outlier data — different component needed for outlier table

### Integration Points
- **CSV export endpoint**: New Express route to generate and return CSV file (stream response with Content-Disposition header)
- **Per-column scores computation**: Backend worker needs to compute and store per-column contributions when job completes (store in Firestore or separate GCS JSON file)
- **GCS lifecycle rules**: Apply to bucket with 7-day age condition and Delete action
- **Job cancellation cleanup**: Extend DELETE /api/jobs/:id to call GCS deleteFile and Vertex AI jobs.cancel APIs
- **Firestore schema update**: Add `filesExpired: boolean` field to jobs collection to track GCS deletion state

</code_context>

<specifics>
## Specific Ideas

- Per-column contribution scores should help researchers answer: "Was this outlier flagged because of weird responses to Question 5, or a pattern across multiple questions?"
- GitHub PR workflow documentation should reference how IliasTriant currently structures PRs (if they have existing patterns) — check closed PRs for examples
- Delete button should be visually distinct from download button (different color, secondary vs primary style) to prevent accidental deletion when user wants to download

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Job cancellation cleanup scope (GCS + Vertex AI) was not discussed in detail but is part of phase requirements.

</deferred>

---

*Phase: 04-operational-features*
*Context gathered: 2026-04-06*
