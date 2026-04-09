# Phase 4: Operational Features - Research

**Researched:** 2026-04-06
**Domain:** CSV export security, GCS lifecycle management, Vertex AI job control, per-column anomaly attribution
**Confidence:** HIGH

## Summary

This phase adds production-grade operational features: CSV export with formula injection protection, automatic file cleanup via GCS lifecycle rules, complete job cancellation (GCS + Vertex AI cleanup), and per-column outlier contribution scores for result interpretation. Additionally, maintainer learns GitHub collaboration workflow to work effectively with IliasTriant.

The technical stack is well-established: Express.js streaming responses with Content-Disposition headers for CSV downloads, GCS lifecycle rules with Delete action and Age conditions, Vertex AI CustomJobs.cancel API for job termination, and decomposition of the existing per-attribute categorical crossentropy loss for contribution scoring. The project already uses shadcn/ui components (Collapsible, Progress, AlertDialog) needed for expandable row UI.

**Primary recommendation:** Use fast-csv for streaming CSV generation (performance-optimized for large datasets), implement single-quote prefixing for formula injection prevention (OWASP standard), configure GCS lifecycle rules with 7-day Age condition via gcloud CLI, extend existing DELETE /api/jobs/:id endpoint to call bucket.file().delete() and aiplatform CustomJob cancel API, decompose VAE.reconstruction_loss() per-attribute into contribution percentages stored in Firestore, and follow established conventional commit format (type(scope): description).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **CSV Export Location & Trigger**: Download button appears on /job/:id progress page when job completes (not upload page, not both locations). Button enabled only when job status is "complete" (not queued/processing/error).

- **CSV Export Content**: Export outlier rows + scores only (not full dataset, not summary-only). Columns: all original data columns from uploaded CSV + `outlier_score` column. Non-outlier rows excluded from export.

- **CSV Export Security**: Prevent Excel formula injection by prefixing dangerous characters (`=`, `+`, `-`, `@` at start of cell). Mitigation: prepend single quote (`'`) to cells starting with dangerous chars. Preserves data readability (e.g., `=-5` becomes `'=-5`, Excel displays as text).

- **Per-Column Contribution Scores Display**: Expandable row detail pattern (click outlier row to expand inline panel). Similar to GitHub PR file list or Slack thread expansion. Expanded panel shows horizontal bar chart of per-column contributions. All columns shown, sorted by contribution (high to low). Not limited to top 5 or top 10.

- **Per-Column Contribution Scores Format**: Horizontal bar chart using shadcn/ui Progress component. Visual bars showing relative contribution (like GitHub commit stats). Each bar labeled with column name + contribution percentage. Color-coded: high contribution (red/orange), medium (yellow), low (gray/green). Sorted descending by contribution value.

- **File Retention Policy**: 7-day retention for all GCS files (uploads and results). Single retention period, not different for uploads vs results. GCS lifecycle rules automatically delete files after 7 days. Firestore job metadata persists even after GCS files deleted.

- **Manual File Deletion**: Delete button on /job/:id page for manual cleanup before 7-day expiration. Separate from job cancellation (cancellation is for running jobs, deletion is for completed jobs). Delete action removes: uploaded CSV from GCS, result files from GCS, keeps Firestore metadata. Confirmation dialog before deletion (prevent accidental data loss).

- **Expired Job Handling**: After GCS files auto-delete (7 days), Firestore job record remains. User accessing /job/:id for expired job sees: job metadata (filename, upload date, status) + "Files expired - data deleted after 7 days" message. No download button shown for expired jobs. No automatic Firestore cleanup — job history preserved indefinitely.

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

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| OPS-01 | User can export outlier results as CSV file | Express streaming + fast-csv + Content-Disposition headers |
| OPS-02 | CSV export prevents formula injection (sanitizes =, +, -, @, \t, \r characters) | Single-quote prefixing (OWASP standard) |
| OPS-03 | CSV export includes proper Content-Disposition headers | res.attachment() or res.setHeader() patterns |
| OPS-04 | Job cancellation deletes GCS files for canceled jobs | bucket.file(path).delete() SDK method |
| OPS-05 | Job cancellation cancels Vertex AI job if running | aiplatform CustomJob cancel API (async best-effort) |
| OPS-06 | Job cancellation updates Firestore status to "canceled" | Existing transaction pattern from worker.py |
| OPS-07 | GCS lifecycle rules delete old uploaded files (7-day retention per user decision) | Lifecycle Delete action with Age: 7 condition |
| OPS-08 | GCS lifecycle rules delete old result files (7-day retention per user decision) | Same lifecycle rule applies to entire bucket |
| OPS-09 | User can see per-column outlier contribution scores in results | Decompose VAE.reconstruction_loss per-attribute |
| OPS-10 | Per-column scores show which survey questions were anomalous | Per-attribute loss × 100 / total_loss = contribution % |
| OPS-11 | User can download failed-rows CSV with specific error descriptions | Similar CSV export with error column (deferred — not in phase scope per CONTEXT.md) |
| OPS-12 | Row-level validation errors indicate encoding issues, missing values, schema mismatches | Worker already validates (WORK-09, WORK-10) — enhance error messages |
| OPS-13 | Signed URLs generated on-demand (1-hour expiration, not 7-day) | Already implemented in jobs.ts (15-min expiration) |
| OPS-14 | Progress tracking writes stage updates to Firestore throughout processing | Worker already implements (Phase 2) — verify completeness |
| GH-01 | Understand PR workflow (branch strategy, naming conventions) | feature/, bugfix/, hotfix/ prefixes + kebab-case |
| GH-02 | Understand commit message conventions (used in this repository) | type(scope): description (feat, fix, docs, chore) |
| GH-03 | Understand code review process (how to request reviews, address feedback) | Standard GitHub PR review workflow |
| GH-04 | Analyze IliasTriant's PR patterns (structure, descriptions, commits) | PR #9, #10 examples show clear descriptions + atomic commits |
| GH-05 | Practice creating well-structured PRs for v1.0 features | Follow conventional commit + squash merge pattern |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fast-csv | Latest (4.x) | CSV generation/parsing | Performance-optimized for large datasets, streaming support, widely used in Node.js ecosystem |
| @google-cloud/storage | 7.18.0 (already installed) | GCS file operations | Official Google SDK, used throughout project |
| @google-cloud/aiplatform | Latest | Vertex AI job control | Official SDK for CustomJob.cancel() |
| shadcn/ui Collapsible | 1.1.12 (installed) | Expandable row UI | Radix UI primitive with ARIA semantics |
| shadcn/ui Progress | 1.1.8 (installed) | Contribution score bars | Visual progress/contribution display |
| shadcn/ui AlertDialog | 1.1.15 (installed) | Delete confirmation | Accessible confirmation dialogs |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Express streaming | 5.2.1 (installed) | Response streaming | Large CSV downloads (> 1MB) |
| TanStack Query | 5.90.12 (installed) | CSV download trigger | Mutation for download action |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| fast-csv | Papa Parse (client-side lib) | Papa Parse optimized for browser, not Node.js streaming; lacks robust server-side docs |
| fast-csv | json2csv | json2csv lacks streaming support, poor performance on large datasets |
| fast-csv | csv-stringify (node-csv) | csv-stringify more verbose API, fast-csv simpler for common cases |
| Single-quote prefix | Tab character prefix | Tab character (\t) escaping more complex, single-quote simpler and OWASP-recommended |

**Installation:**
```bash
npm install fast-csv @google-cloud/aiplatform
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/server/
├── routes/
│   └── jobs.ts              # Add GET /jobs/:id/export endpoint
├── utils/
│   ├── fileValidation.ts    # Existing validation
│   └── csvSanitization.ts   # NEW: Formula injection prevention
└── services/
    └── vertexAi.ts          # NEW: Vertex AI job cancellation helper

frontend/client/
├── components/
│   └── results/
│       ├── OutlierTable.tsx         # NEW: Table with expandable rows
│       └── ContributionScores.tsx   # NEW: Per-column bar chart
└── hooks/
    └── useJobCancellation.ts        # EXTEND: Add GCS + Vertex cleanup

worker.py                    # EXTEND: Compute per-column contributions

.planning/docs/
└── GITHUB-WORKFLOW.md       # NEW: PR workflow documentation
```

### Pattern 1: CSV Export with Formula Injection Protection
**What:** Streaming CSV generation with automatic sanitization of dangerous formula characters
**When to use:** Any CSV export containing user-provided data
**Example:**
```typescript
// Source: OWASP CSV Injection guide + fast-csv docs
import { format } from 'fast-csv';

// Sanitization function
function sanitizeFormulaInjection(value: string): string {
  if (typeof value !== 'string') return value;

  const dangerousChars = ['=', '+', '-', '@', '\t', '\r'];
  if (dangerousChars.some(char => value.startsWith(char))) {
    return `'${value}`; // Prefix with single quote
  }
  return value;
}

// Express route
router.get('/jobs/:id/export', requireAuth, downloadLimiter, async (req, res) => {
  const { id } = req.params;

  // Fetch outlier data from Firestore
  const job = await firestore.collection('jobs').doc(id).get();
  const outliers = job.data().outliers; // Array of row objects

  // Set headers for CSV download
  res.attachment(`outliers-${id}.csv`);
  res.setHeader('Content-Type', 'text/csv');

  // Stream CSV with sanitization
  const csvStream = format({ headers: true });
  csvStream.pipe(res);

  outliers.forEach(row => {
    const sanitizedRow = Object.fromEntries(
      Object.entries(row).map(([key, value]) => [
        key,
        sanitizeFormulaInjection(String(value))
      ])
    );
    csvStream.write(sanitizedRow);
  });

  csvStream.end();
});
```

### Pattern 2: GCS Lifecycle Rule Configuration
**What:** Automatic deletion of old files using bucket lifecycle management
**When to use:** Any temporary file storage requiring automatic cleanup
**Example:**
```bash
# Source: Google Cloud Storage lifecycle docs
# Apply lifecycle rule to bucket (7-day retention)
gcloud storage buckets update gs://your-bucket-name --lifecycle-file=lifecycle.json

# lifecycle.json
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 7
        }
      }
    ]
  }
}
```

**Alternative: Node.js SDK approach:**
```typescript
// Source: @google-cloud/storage docs
import { Storage } from '@google-cloud/storage';

const storage = new Storage();
const bucket = storage.bucket(BUCKET_NAME);

await bucket.addLifecycleRule({
  action: { type: 'delete' },
  condition: { age: 7 }
});

await bucket.setMetadata({
  lifecycle: {
    rule: [{
      action: { type: 'Delete' },
      condition: { age: 7 }
    }]
  }
});
```

### Pattern 3: Vertex AI Job Cancellation
**What:** Cancel running Vertex AI CustomContainerTrainingJob
**When to use:** User-initiated job cancellation, cleanup on errors
**Example:**
```typescript
// Source: Google Vertex AI Node.js SDK docs
import { JobServiceClient } from '@google-cloud/aiplatform';

async function cancelVertexAIJob(projectId: string, location: string, jobId: string) {
  const client = new JobServiceClient({
    apiEndpoint: `${location}-aiplatform.googleapis.com`
  });

  const name = `projects/${projectId}/locations/${location}/customJobs/${jobId}`;

  try {
    await client.cancelCustomJob({ name });
    logger.info(`Vertex AI job ${jobId} cancellation requested`);
    // Note: Cancellation is asynchronous and best-effort (not guaranteed)
  } catch (error) {
    logger.warn(`Failed to cancel Vertex AI job: ${error.message}`);
    // Job may already be complete or not exist
  }
}
```

### Pattern 4: Per-Column Contribution Score Computation
**What:** Decompose total reconstruction loss into per-attribute contributions
**When to use:** Outlier result display, feature importance analysis
**Example:**
```python
# Source: Existing model/base.py VAE.reconstruction_loss + evaluate/outliers.py
def compute_per_column_contributions(
    data,
    predictions,
    attr_cardinalities,
    column_names
):
    """
    Decompose reconstruction loss into per-column contribution percentages.

    Returns: List of (column_name, contribution_percentage) tuples
    """
    per_attr_losses = []
    start_idx = 0

    for categories in attr_cardinalities:
        x_attr = data[:, start_idx : start_idx + categories]
        y_attr = predictions[:, start_idx : start_idx + categories]

        # Categorical crossentropy per attribute
        attr_loss = tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
        attr_loss = attr_loss / np.log(categories)  # Normalize by cardinality

        per_attr_losses.append(attr_loss.numpy())
        start_idx += categories

    # Convert to contribution percentages
    total_loss = sum(per_attr_losses)
    contributions = [
        (column_names[i], (loss / total_loss) * 100 if total_loss > 0 else 0)
        for i, loss in enumerate(per_attr_losses)
    ]

    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x[1], reverse=True)

    return contributions
```

### Pattern 5: Expandable Table Row with shadcn/ui
**What:** Table with collapsible row details using Collapsible component
**When to use:** Detailed information hidden by default, expand on click
**Example:**
```tsx
// Source: shadcn/ui Collapsible docs + community patterns
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import { ChevronDown, ChevronRight } from "lucide-react";

function OutlierRow({ outlier, contributions }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="border-b">
        {/* Main row */}
        <div className="flex items-center p-4 hover:bg-slate-50 cursor-pointer">
          <CollapsibleTrigger className="flex items-center gap-2">
            {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            <span className="font-medium">Score: {outlier.score.toFixed(2)}</span>
          </CollapsibleTrigger>
          <div className="ml-auto">{outlier.id}</div>
        </div>

        {/* Expanded details */}
        <CollapsibleContent className="px-4 pb-4 bg-slate-50">
          <h4 className="font-semibold mb-2">Per-Column Contributions</h4>
          <div className="space-y-2">
            {contributions.map(([column, percentage]) => (
              <div key={column} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>{column}</span>
                  <span className="font-medium">{percentage.toFixed(1)}%</span>
                </div>
                <Progress
                  value={percentage}
                  className={getContributionColor(percentage)}
                />
              </div>
            ))}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function getContributionColor(percentage: number): string {
  if (percentage > 20) return "bg-red-500";      // High contribution
  if (percentage > 10) return "bg-orange-500";   // Medium-high
  if (percentage > 5) return "bg-yellow-500";    // Medium
  return "bg-gray-400";                          // Low contribution
}
```

### Pattern 6: Conventional Commit Messages (Project Standard)
**What:** Structured commit messages following conventional commit format
**When to use:** All commits in this repository
**Example:**
```bash
# Source: Repository commit history analysis

# Format: type(scope): description
# Types: feat, fix, docs, chore, refactor, test, style, perf
# Scopes: phase number (01, 02, 03, 04) or plan (01-01, 03-03A)

# Good examples from repository:
feat(03-03A): create useJobPolling hook with TanStack Query
fix(03-01): add missing @radix-ui/react-collapsible dependency
docs(04): capture phase context
chore(03-04A): enable noImplicitAny in tsconfig.json

# For this phase:
feat(04-01): add CSV export endpoint with formula injection protection
fix(04-02): cancel Vertex AI job on user cancellation
docs(04): add GitHub PR workflow guide
```

### Anti-Patterns to Avoid
- **Building custom CSV parser/generator:** Use fast-csv — handles edge cases (quoted fields, newlines in values, encoding)
- **Blocking CSV generation in memory:** Stream large CSVs directly to response — prevents memory overflow
- **Assuming Vertex AI cancellation is synchronous:** Cancel API is async and best-effort — job may complete before cancellation
- **Hardcoding lifecycle rules in code:** Use gcloud CLI or Terraform for infrastructure config — easier to audit and version
- **Computing contributions client-side:** Backend computes and stores in Firestore — avoids sending full model predictions to browser

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CSV generation | Custom CSV string builder | fast-csv library | Handles quoted fields, newlines in cells, encoding, streaming |
| Formula injection protection | Complex regex sanitizer | Single-quote prefix for dangerous chars | OWASP-recommended, simple, preserves readability |
| File lifecycle management | Cron job to delete old files | GCS lifecycle rules | Cloud-native, reliable, no maintenance, automatic |
| Vertex AI job tracking | Custom job status poller | Vertex AI job state API | Official SDK handles retries, exponential backoff |
| CSV streaming | Manual chunk writing | Express res.pipe() + fast-csv | Handles backpressure, memory management |

**Key insight:** File operations, CSV handling, and cloud service integrations have mature libraries that handle edge cases better than custom code. Formula injection prevention is deceptively simple (single-quote prefix) — don't overcomplicate.

## Common Pitfalls

### Pitfall 1: Formula Injection via Tab/Newline Characters
**What goes wrong:** Sanitizing only `=+-@` but missing `\t` and `\r` allows formula injection via tab-prefixed cells
**Why it happens:** Most CSV injection guides focus on visible characters, but Excel interprets tabs and carriage returns as formula prefixes
**How to avoid:** Include `\t` and `\r` in dangerous character list (per OPS-02 requirement)
**Warning signs:** Security scan tools flag tab-prefixed fields; manual testing in Excel shows formula execution

### Pitfall 2: GCS Lifecycle Rules Take 24 Hours to Apply
**What goes wrong:** Applying lifecycle rule and expecting immediate deletion — files persist for up to 24 hours
**Why it happens:** GCS evaluates lifecycle rules once per day, changes take time to propagate
**How to avoid:** Document delay in user-facing messages; use manual delete for immediate cleanup
**Warning signs:** Files still present after lifecycle rule applied; user confusion about retention policy

### Pitfall 3: Vertex AI Cancellation is Best-Effort, Not Guaranteed
**What goes wrong:** Assuming `cancelCustomJob()` immediately stops the job — job may complete before cancellation
**Why it happens:** Vertex AI cancellation is asynchronous; job state transitions (training → stopping → canceled) take time
**How to avoid:** Update Firestore to "canceling" state immediately, poll job status to confirm cancellation, handle race condition if job completes before cancel
**Warning signs:** Job marked "canceled" but billing shows full training time; job produces results after cancellation

### Pitfall 4: Per-Column Contributions Don't Sum to 100% (Floating Point)
**What goes wrong:** Displaying per-column contributions that sum to 99.8% or 100.3% due to rounding errors
**Why it happens:** Dividing loss by total_loss and converting to percentage accumulates floating-point errors
**How to avoid:** Normalize contributions to exactly 100% after rounding, or show "~100%" disclaimer
**Warning signs:** User questions about math; QA finds contribution sums inconsistent

### Pitfall 5: Exporting Full Dataset Instead of Outliers Only
**What goes wrong:** CSV export includes all rows (outliers + non-outliers), file size explodes
**Why it happens:** Misunderstanding OPS-01 requirement — export should be outliers only (per CONTEXT.md)
**How to avoid:** Filter to rows where `outlier_score > threshold` before CSV generation; verify file size reasonable
**Warning signs:** 50MB download for dataset with 5% outliers (should be ~2.5MB)

### Pitfall 6: Delete Button Available on Expired Jobs
**What goes wrong:** Showing delete button on jobs where GCS files already auto-deleted (nothing to delete)
**Why it happens:** Not checking `filesExpired` flag before rendering delete button
**How to avoid:** Conditional rendering: delete button only if `!job.filesExpired && job.status === 'complete'`
**Warning signs:** User clicks delete on expired job, gets error "File not found"

### Pitfall 7: Committing Without Conventional Commit Format
**What goes wrong:** Commit messages like "fixed stuff" or "updates" don't follow project standard
**Why it happens:** Not reviewing repository commit history before contributing
**How to avoid:** Use `git log --oneline` to review format, configure commit message template, use commitlint tool
**Warning signs:** PR review requests commit message changes; CI fails on commit message lint

## Code Examples

Verified patterns from official sources:

### Express CSV Download with Content-Disposition
```typescript
// Source: Express.js res.attachment() docs
import express from 'express';
import { format } from 'fast-csv';

router.get('/jobs/:id/export', requireAuth, downloadLimiter, async (req, res) => {
  const { id } = req.params;

  // Fetch data from Firestore
  const job = await firestore.collection('jobs').doc(id).get();
  if (!job.exists || job.data().status !== 'complete') {
    return res.status(404).json({ error: 'Job not found or not complete' });
  }

  const outliers = job.data().outliers || [];

  // Set headers for download
  res.attachment(`outliers-${id}.csv`); // Sets Content-Disposition + Content-Type

  // Stream CSV
  const csvStream = format({ headers: true });
  csvStream.pipe(res);

  outliers.forEach(row => csvStream.write(row));
  csvStream.end();
});
```

### GCS File Deletion
```typescript
// Source: @google-cloud/storage SDK docs
import { storage } from '../config/gcp-clients';

async function deleteJobFiles(bucketName: string, uploadPath: string, resultPath?: string) {
  const bucket = storage.bucket(bucketName);

  // Delete uploaded CSV
  await bucket.file(uploadPath).delete();
  logger.info(`Deleted upload file: ${uploadPath}`);

  // Delete result files if exist
  if (resultPath) {
    await bucket.file(resultPath).delete();
    logger.info(`Deleted result file: ${resultPath}`);
  }
}
```

### Vertex AI Job Cancellation (Python Worker)
```python
# Source: Google Cloud aiplatform Python SDK
from google.cloud import aiplatform

def cancel_vertex_job(project_id: str, location: str, job_display_name: str):
    """Cancel a running Vertex AI CustomContainerTrainingJob."""
    aiplatform.init(project=project_id, location=location)

    # List jobs with matching display name
    jobs = aiplatform.CustomContainerTrainingJob.list(
        filter=f'display_name="{job_display_name}"'
    )

    for job in jobs:
        if job.state in ['RUNNING', 'PENDING']:
            try:
                job.cancel()
                logger.info(f"Canceled Vertex AI job: {job.resource_name}")
            except Exception as e:
                logger.warning(f"Failed to cancel job: {e}")
                # Job may have completed already
```

### Per-Column Contribution Storage in Firestore
```python
# Source: Project model/base.py + Firestore SDK
def store_outlier_contributions(job_id: str, outliers_with_contributions: list):
    """
    Store outlier results with per-column contribution scores.

    Args:
        job_id: Firestore job document ID
        outliers_with_contributions: List of dicts with {row_id, score, contributions}
    """
    job_ref = db.collection('jobs').document(job_id)

    # Transform contributions for Firestore storage
    outliers = []
    for outlier in outliers_with_contributions:
        outliers.append({
            'rowId': outlier['row_id'],
            'score': float(outlier['score']),
            'contributions': [
                {'column': col, 'percentage': float(pct)}
                for col, pct in outlier['contributions']
            ]
        })

    job_ref.update({
        'outliers': outliers,
        'status': 'complete',
        'completedAt': firestore.SERVER_TIMESTAMP
    })
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual file cleanup cron jobs | GCS lifecycle rules | 2020+ | Cloud-native, no maintenance, audit logs |
| Synchronous CSV generation in memory | Streaming with fast-csv/csv-stringify | 2018+ | Handles multi-GB files without memory crash |
| Tab-character formula injection mitigation | Single-quote prefix | 2021+ (OWASP update) | Simpler implementation, better compatibility |
| Total outlier score only | Per-feature contribution scores | Research ongoing | Explainability for users ("why is this an outlier?") |
| Vertex AI job polling for cancellation | Direct cancel API call | 2022+ (Vertex AI GA) | Immediate cancellation request, cleaner code |

**Deprecated/outdated:**
- **Papa Parse for Node.js CSV generation:** Papa Parse documentation focuses on browser usage, lacks robust Node.js streaming examples; fast-csv is Node.js-native
- **json2csv library:** Lacks streaming support (entire dataset in memory), poor performance on large files
- **Firestore client SDK v7:** Project uses v8 (breaking changes in transaction API)

## Open Questions

1. **Vertex AI job resource name format**
   - What we know: Jobs created with display_name=`autoencoder-{job_id}`, need resource_name for cancel API
   - What's unclear: Whether to store resource_name in Firestore during job creation, or query by display_name for cancellation
   - Recommendation: Store resource_name in Firestore when job is created (process_upload_vertex) — avoids list query overhead during cancellation

2. **Per-column contribution computation performance**
   - What we know: Need to decompose VAE.reconstruction_loss per-attribute (model/base.py pattern exists)
   - What's unclear: Whether to compute during training (store in model) or post-hoc (compute in worker when job completes)
   - Recommendation: Compute post-hoc in worker after prediction — keeps model unchanged, easier to iterate on contribution algorithm

3. **GCS lifecycle rule scope**
   - What we know: 7-day retention for all files, applied at bucket level
   - What's unclear: Whether to use prefix matching (uploads/ and results/ separately) or single rule for entire bucket
   - Recommendation: Single rule for entire bucket (simpler) — all files temporary, no reason to differentiate

4. **Expired job download button behavior**
   - What we know: After 7 days, GCS files deleted but Firestore metadata persists
   - What's unclear: How to track `filesExpired` state (cron job to check GCS, rely on lifecycle event, client-side Age check)
   - Recommendation: Client-side check: if `createdAt + 7 days < now`, hide download button — no server-side state management needed

## Validation Architecture

> nyquist_validation is not explicitly set to false in .planning/config.json, so this section is included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Jest 30.3.0 (Node.js tests) |
| Config file | frontend/package.json (test script configured) |
| Quick run command | `npm test -- --testPathPattern=csvSanitization` |
| Full suite command | `npm test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OPS-01 | CSV export returns file with outlier data | integration | `npm test -- --testPathPattern=csvExport` | ❌ Wave 0 |
| OPS-02 | Formula injection chars prefixed with quote | unit | `npm test -- --testPathPattern=csvSanitization` | ❌ Wave 0 |
| OPS-03 | Content-Disposition header set correctly | integration | `npm test -- --testPathPattern=csvExport` | ❌ Wave 0 |
| OPS-04 | Job cancellation deletes GCS files | integration | `npm test -- --testPathPattern=jobCancellation` | ❌ Wave 0 |
| OPS-05 | Vertex AI job cancel API called | unit (mock) | `npm test -- --testPathPattern=vertexAi` | ❌ Wave 0 |
| OPS-06 | Firestore status updated to "canceled" | integration | `npm test -- --testPathPattern=jobCancellation` | ❌ Wave 0 |
| OPS-07 | GCS lifecycle rule applied (7-day age) | manual-only | N/A (infrastructure config) | N/A |
| OPS-08 | GCS lifecycle rule applies to all files | manual-only | N/A (same as OPS-07) | N/A |
| OPS-09 | Per-column contributions computed | unit | `pytest tests/test_contributions.py::test_compute_contributions -x` | ❌ Wave 0 |
| OPS-10 | Contributions sum to ~100% | unit | `pytest tests/test_contributions.py::test_contribution_sum -x` | ❌ Wave 0 |
| OPS-11 | Failed-rows CSV export (deferred) | N/A | N/A | N/A |
| OPS-12 | Enhanced error messages (existing) | unit | `pytest tests/test_validation.py -x` | ✅ |
| OPS-13 | Signed URL expiration (existing) | unit | `npm test -- --testPathPattern=jobs` | ✅ |
| OPS-14 | Progress updates (existing) | integration | `pytest tests/test_worker.py -x` | ✅ |
| GH-01 | Branch naming conventions | manual-only | N/A (documentation/practice) | N/A |
| GH-02 | Commit message conventions | manual-only | N/A (documentation/practice) | N/A |
| GH-03 | PR review process | manual-only | N/A (documentation/practice) | N/A |
| GH-04 | IliasTriant PR analysis | manual-only | N/A (documentation/practice) | N/A |
| GH-05 | PR creation practice | manual-only | N/A (documentation/practice) | N/A |

### Sampling Rate
- **Per task commit:** `npm test -- --testPathPattern=<relevant>` (targeted tests for changed code)
- **Per wave merge:** `npm test && pytest tests/` (full suite)
- **Phase gate:** Full suite green + manual verification of GCS lifecycle rule + PR workflow documentation complete

### Wave 0 Gaps
- [ ] `frontend/server/__tests__/utils/csvSanitization.test.ts` — covers OPS-02
- [ ] `frontend/server/__tests__/routes/csvExport.test.ts` — covers OPS-01, OPS-03
- [ ] `frontend/server/__tests__/routes/jobCancellation.test.ts` — covers OPS-04, OPS-06
- [ ] `frontend/server/__tests__/services/vertexAi.test.ts` — covers OPS-05
- [ ] `tests/test_contributions.py` — covers OPS-09, OPS-10
- [ ] Framework install: `npm install --save-dev @types/fast-csv` — TypeScript types for fast-csv

## Sources

### Primary (HIGH confidence)
- [OWASP CSV Injection](https://owasp.org/www-community/attacks/CSV_Injection) - Formula injection attack vectors and prevention
- [Google Cloud Storage Lifecycle Docs](https://docs.cloud.google.com/storage/docs/lifecycle) - Lifecycle rule configuration and evaluation
- [Vertex AI Cancel Custom Job](https://docs.cloud.google.com/vertex-ai/docs/samples/aiplatform-cancel-custom-job-sample) - Job cancellation API
- [Express.js API Docs](https://expressjs.com/en/api.html) - res.attachment() and streaming responses
- [@google-cloud/storage SDK](https://www.npmjs.com/package/@google-cloud/storage) - bucket.file().delete() method
- [shadcn/ui Collapsible](https://ui.shadcn.com/docs/components/radix/collapsible) - Expandable row component
- Repository commit history - Conventional commit format analysis

### Secondary (MEDIUM confidence)
- [fast-csv Documentation](https://c2fo.github.io/fast-csv/) - CSV generation and streaming (verified via npm-compare)
- [Papa Parse npm](https://www.npmjs.com/package/papaparse) - CSV parsing capabilities (browser-focused)
- [Better Stack Papa Parse Guide](https://betterstack.com/community/guides/scaling-nodejs/parsing-csv-files-with-papa-parse/) - Node.js usage patterns
- [Practical Autoencoder Anomaly Detection Paper](https://link.springer.com/article/10.1186/s42400-022-00134-9) - Per-feature reconstruction error approach
- [Git Branch Naming Conventions Guide](https://medium.com/@regondaakhil/best-practices-for-git-branch-naming-conventions-and-pr-creation-on-github-14a451d345dc) - feature/, bugfix/, hotfix/ patterns
- [Conventional Commit PR Naming](https://namingconvention.org/git/pull-request-naming) - PR title format standards

### Tertiary (LOW confidence)
- [csv-parser vs json2csv comparison](https://npm-compare.com/csv-parser,csv-writer,json2csv) - Library benchmarks (no specific 2026 data)
- [DeepWiki Reconstruction Error](https://deepwiki.com/vincrichard/LSTM-AutoEncoder-Unsupervised-Anomaly-Detection/6.1-reconstruction-error-calculation) - General reconstruction error concepts

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Libraries well-documented, widely used, already installed (shadcn/ui, GCS SDK)
- CSV export security: HIGH - OWASP standard, simple implementation, verified in production systems
- GCS lifecycle rules: HIGH - Official Google Cloud docs, stable API since 2018
- Vertex AI cancellation: MEDIUM - API documented but async behavior requires testing
- Per-column contributions: MEDIUM - Pattern exists in codebase (model/base.py), need to adapt for outlier scoring
- GitHub workflow: HIGH - Repository commit history provides clear examples

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (30 days - stable technologies)

## Unresolved Questions

1. **Should we store Vertex AI job resource_name in Firestore during job creation?**
   - Impact: Affects job cancellation implementation complexity
   - Options: Store resource_name vs query by display_name
   - Recommendation: Store resource_name (simpler cancellation logic)

2. **How to track filesExpired state in Firestore?**
   - Impact: Affects expired job UI behavior
   - Options: Client-side age check vs server-side cron vs GCS event trigger
   - Recommendation: Client-side age check (no server state needed)

3. **Should per-column contributions be normalized to exactly 100%?**
   - Impact: User experience and data accuracy
   - Options: Show raw percentages (may sum to 99.8%) vs normalize vs disclaimer
   - Recommendation: Normalize to 100% after rounding (better UX)
