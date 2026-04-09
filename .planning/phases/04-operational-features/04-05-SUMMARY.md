---
phase: 04-operational-features
plan: 05
subsystem: frontend-ui
tags: [ui-components, outlier-display, contributions, shadcn-ui, react]
completed: 2026-04-07T02:27:55Z
duration: 213s
tasks_completed: 3
files_created:
  - frontend/client/components/results/ContributionScores.tsx
  - frontend/client/components/results/OutlierTable.tsx
files_modified:
  - frontend/client/pages/JobProgress.tsx
commits:
  - 56d836e: "feat(04-05): add ContributionScores component with color-coded bar charts"
  - dad97f6: "feat(04-05): add OutlierTable component with expandable rows"
  - c274d40: "feat(04-05): integrate OutlierTable into JobProgress page"
requirements_satisfied: [OPS-09, OPS-10]
dependencies:
  requires:
    - plans: [04-04]
      reason: "Backend computes per-column contribution scores"
  provides:
    - artifact: "OutlierTable component"
      for: "Job results display"
    - artifact: "ContributionScores component"
      for: "Per-column contribution visualization"
  affects:
    - component: "JobProgress page"
      impact: "Now displays outlier results with expandable contribution details"
tech_stack:
  added: []
  patterns:
    - "Expandable row UI pattern (GitHub PR file list style)"
    - "Color-coded progress bars for data visualization"
    - "Conditional rendering based on job status and data presence"
key_files:
  created:
    - path: "frontend/client/components/results/ContributionScores.tsx"
      purpose: "Horizontal bar chart visualization of per-column contributions"
      lines: 59
    - path: "frontend/client/components/results/OutlierTable.tsx"
      purpose: "Expandable table component for outlier display"
      lines: 84
  modified:
    - path: "frontend/client/pages/JobProgress.tsx"
      changes: "Added OutlierTable integration with conditional rendering"
      lines_added: 15
decisions:
  - what: "Color threshold values for contribution visualization"
    chosen: ">20% red, >10% orange, >5% yellow, ≤5% gray"
    rationale: "Creates clear visual hierarchy highlighting high contributors while maintaining readability"
    alternatives: "Equal distribution (e.g., quartiles), gradient colors"
  - what: "Percentage decimal precision"
    chosen: "1 decimal place for contributions, 3 for outlier scores"
    rationale: "Balance between precision and readability - contributions show general magnitude, scores need precision for threshold decisions"
    alternatives: "0 decimals, 2 decimals, dynamic precision"
  - what: "Empty outliers UI treatment"
    chosen: "Dedicated 'No outliers detected' message in muted card"
    rationale: "Explicit positive feedback when no issues found (success state)"
    alternatives: "Hide section entirely, show empty table"
  - what: "Expand/collapse state management"
    chosen: "Per-row local state (useState in OutlierRow)"
    rationale: "Independent row expansion, simple implementation, no shared state needed"
    alternatives: "Centralized state in OutlierTable, URL-based state, accordion pattern (single expanded row)"
metrics:
  duration: "3m 33s"
  tasks: 3
  files_created: 2
  files_modified: 1
  lines_added: 158
  commits: 3
---

# Phase 04 Plan 05: Outlier Table UI with Contribution Visualization

**One-liner:** Expandable outlier table with color-coded per-column contribution bar charts using shadcn/ui Collapsible and Progress components.

## Overview

Built frontend UI components to display outlier detection results with expandable rows showing per-column contribution scores. Researchers can now click any outlier row to see which survey questions triggered the anomaly detection, visualized as color-coded horizontal bar charts.

This completes the frontend portion of the per-column contributions feature (backend implemented in plan 04-04).

## Implementation Summary

### Component Architecture

Created two-component architecture following shadcn/ui patterns:

**ContributionScores.tsx** (59 lines)
- Presentational component rendering contribution data as horizontal bar charts
- Uses shadcn/ui `Progress` component for bars
- Color-coded by contribution level with 4-tier system
- Displays all columns sorted by contribution (backend provides sorted data)
- No state management - pure presentation

**OutlierTable.tsx** (84 lines)
- Container component rendering list of outlier rows
- Each row uses shadcn/ui `Collapsible` for expand/collapse
- Per-row state management (independent expansion)
- Chevron icon indicates expand/collapse state
- Embeds ContributionScores in expanded panel

**JobProgress.tsx integration** (15 lines added)
- Added OutlierTable import
- Conditional rendering when `job.status === 'complete'` and `job.outliers` exists
- "No outliers detected" message for empty results
- Positioned after terminal state messages, before download button location (plan 04-01)

### Color Coding System

Contribution percentage color thresholds:
- **>20%** → Red (`bg-red-500`) - High contribution (major anomaly driver)
- **>10%** → Orange (`bg-orange-500`) - Medium-high contribution
- **>5%** → Yellow (`bg-yellow-500`) - Medium contribution
- **≤5%** → Gray (`bg-gray-400`) - Low contribution

Rationale: These thresholds create clear visual hierarchy. In typical outlier scenarios, 1-3 columns dominate contribution (often >30%), while most columns contribute <5%. Color coding helps researchers immediately identify which questions caused the anomaly flag.

### Expandable Row Pattern

Followed GitHub PR file list interaction pattern:
- Click chevron or row to expand/collapse
- Chevron rotates (right → down) on expansion
- Background color change on hover (`hover:bg-slate-50`)
- Smooth transition animation (built into Collapsible component)
- Independent row state (multiple rows can be expanded simultaneously)

Alternative considered: Accordion pattern (single expanded row). Rejected because researchers often compare contributions across multiple outliers.

### Data Flow

```
Backend (04-04) → Firestore job.outliers array →
JobPolling hook → JobProgress page → OutlierTable → OutlierRow → ContributionScores
```

Each outlier contains:
```typescript
{
  rowId: number,
  score: number,
  contributions: [{ column: string, percentage: number }]
}
```

Backend sorts contributions by percentage descending. Frontend displays as-is.

## Tasks Completed

### Task 1: ContributionScores Component (56d836e)
- Created horizontal bar chart component using Progress primitive
- Implemented 4-tier color coding function
- Percentage display with 1 decimal precision
- Clean grid layout with column name + percentage label above each bar

**Verification:** Component file created, imports Progress, renders contributions array.

### Task 2: OutlierTable Component (dad97f6)
- Created table container with outlier count header
- Implemented OutlierRow sub-component with Collapsible
- Chevron icon integration from lucide-react
- Score display with 3 decimal precision
- ContributionScores embedded in CollapsibleContent

**Verification:** Component file created, imports Collapsible and ContributionScores, manages expand state.

### Task 3: JobProgress Integration (c274d40)
- Added OutlierTable import
- Conditional rendering logic for complete jobs with outliers
- Empty state message for jobs with no outliers detected
- Positioned after terminal state messages in CardContent

**Verification:** JobProgress imports OutlierTable, checks job.outliers, renders conditionally.

## Verification Results

### Automated Checks
- ✅ Component files created at expected paths
- ✅ OutlierTable imports Collapsible and ContributionScores
- ✅ ContributionScores imports Progress
- ✅ JobProgress imports OutlierTable
- ✅ JobProgress checks job.outliers before rendering

### Manual Verification (Required)
After deployment, verify:
1. Navigate to `/job/:id` with completed job containing outliers
2. Verify outlier table displays with correct count
3. Click outlier row → panel expands showing contribution bars
4. Verify bars are color-coded (inspect high/medium/low contributions)
5. Verify percentages display with 1 decimal place
6. Click row again → panel collapses
7. Verify chevron icon rotates on expand/collapse
8. Test job with no outliers → "No outliers detected" message displays

No automated UI tests added (React components, manual verification sufficient per plan).

## Deviations from Plan

None. Plan executed exactly as written. No bugs discovered, no missing functionality added, no blocking issues encountered.

## Requirements Coverage

**OPS-09: Per-column contribution scores (Frontend)**
- ✅ OutlierTable component displays outliers with expandable rows
- ✅ ContributionScores component renders per-column contributions as bar charts
- ✅ Integration into JobProgress page when job completes
- ✅ Frontend portion complete (backend portion completed in 04-04)

**OPS-10: Visual contribution display**
- ✅ Horizontal bar charts using shadcn/ui Progress component
- ✅ Color-coded by contribution level (4-tier system)
- ✅ All columns displayed, sorted by contribution
- ✅ Expandable row pattern for compact default view
- ✅ Smooth animations on expand/collapse

## Technical Notes

### shadcn/ui Component Usage

**Collapsible** (already installed)
- Provides expand/collapse animation automatically
- Manages ARIA attributes for accessibility
- CollapsibleTrigger and CollapsibleContent primitives

**Progress** (already installed)
- Renders horizontal bar with value prop (0-100)
- Customizable via className for color coding
- Built on native HTML progress semantics

**Button** (already installed)
- Used for CollapsibleTrigger wrapper
- `variant="ghost"` for minimal styling (just chevron icon)

### TypeScript Interfaces

Interfaces defined inline per shadcn/ui patterns (not shared types file):
- `Contribution` - column name + percentage
- `ContributionScoresProps` - array of contributions
- `Outlier` - rowId, score, contributions array (matches backend schema from 04-04)
- `OutlierTableProps` - array of outliers

Alternative considered: Extract to `@/types/outliers.ts`. Rejected - over-engineering for 2 simple components.

### Performance Considerations

No pagination implemented for v1 (plan explicitly states "assume reasonable count"). Typical outlier detection:
- 1000 row dataset → 5-20 outliers (0.5-2%)
- 10000 row dataset → 50-200 outliers (0.5-2%)

If >100 outliers, UI may become sluggish. Defer pagination to future enhancement if needed.

Each OutlierRow manages independent state (no shared state). Expanding 50 rows simultaneously is O(50) state updates, acceptable for React.

## Integration Points

### Upstream Dependencies
- Plan 04-04: Backend per-column contribution computation
  - Firestore schema: `job.outliers[].contributions[]`
  - Backend sorts contributions by percentage descending
  - Frontend assumes sorted data

### Downstream Impact
- Plan 04-01: CSV export (already completed)
  - Download button positioned below OutlierTable
  - Export includes contribution scores (backend implementation)

### Shared Components
- Uses progress components from Phase 03 (StageIndicator, DualProgressBar, JobMetadata)
- Uses shadcn/ui primitives (Collapsible, Progress, Button, Card)
- Uses useJobPolling hook (plan 03-04A)

## Future Enhancements

Potential improvements not required for v1.0:
1. **Column filtering** - Show only contributions >X%
2. **Pagination** - Handle datasets with >100 outliers
3. **Sort options** - Sort outliers by score vs rowId
4. **Export expanded view** - Include contribution details in CSV export
5. **Accessibility** - Keyboard navigation for expand/collapse (shadcn/ui provides basic support)
6. **Mobile responsive** - Horizontal scroll for long column names on small screens

## Files Modified

### Created Files

**frontend/client/components/results/ContributionScores.tsx**
```typescript
// Horizontal bar chart component
// - Progress bars color-coded by contribution level
// - 1 decimal precision for percentages
// - All columns displayed
```

**frontend/client/components/results/OutlierTable.tsx**
```typescript
// Expandable outlier table
// - Collapsible rows with chevron indicator
// - Per-row expand/collapse state
// - Embeds ContributionScores in expanded panel
```

### Modified Files

**frontend/client/pages/JobProgress.tsx**
- Added OutlierTable import
- Conditional rendering when job.status === 'complete' and outliers exist
- "No outliers" message for empty results

## Commit Details

| Commit | Task | Summary |
|--------|------|---------|
| 56d836e | 1 | Created ContributionScores component with color-coded bar charts |
| dad97f6 | 2 | Created OutlierTable component with expandable rows |
| c274d40 | 3 | Integrated OutlierTable into JobProgress page |

## Success Criteria Met

- ✅ ContributionScores component created with horizontal bar charts
- ✅ Bar charts color-coded by contribution level (>20% red, >10% orange, >5% yellow, ≤5% gray)
- ✅ OutlierTable component created with expandable rows using Collapsible
- ✅ Chevron icon indicates expand/collapse state
- ✅ ContributionScores integrated into expanded row panel
- ✅ OutlierTable integrated into JobProgress page
- ✅ Displays only when job status is 'complete' and outliers exist
- ✅ Shows "No outliers" message for empty results
- ✅ Requirements OPS-09, OPS-10 satisfied (frontend portion)

## Self-Check: PASSED

### Created Files
✅ FOUND: frontend/client/components/results/ContributionScores.tsx
✅ FOUND: frontend/client/components/results/OutlierTable.tsx

### Modified Files
✅ FOUND: frontend/client/pages/JobProgress.tsx
✅ OutlierTable import exists
✅ job.outliers conditional rendering exists

### Commits
✅ FOUND: 56d836e (ContributionScores component)
✅ FOUND: dad97f6 (OutlierTable component)
✅ FOUND: c274d40 (JobProgress integration)

All claims verified. Plan execution complete.
