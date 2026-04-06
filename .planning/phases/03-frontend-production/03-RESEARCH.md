# Phase 3: Frontend Production - Research

**Researched:** 2026-04-06
**Domain:** React 18 production UX patterns (error boundaries, progress tracking, polling lifecycle, build tooling)
**Confidence:** HIGH

## Summary

Phase 3 focuses on production-hardening the React 18 frontend with error recovery, multi-stage progress feedback, and build reliability. The phase addresses 22 requirements spanning error boundaries (FE-01 to FE-03), progress/polling UX (FE-04 to FE-10), missing dependencies (FE-11 to FE-15), TypeScript strict mode (FE-16 to FE-17), and infrastructure cleanup (FE-18 to FE-22).

**Key findings:** React error boundaries require class components or the `react-error-boundary` library (v6.1.1, 1889+ dependent projects). TanStack Query's `refetchInterval` with function-based conditional logic provides clean polling cleanup. TypeScript 6.0 assumes `--strict` by default, requiring incremental migration (`noImplicitAny` → `strictNullChecks` → `strict`). Papa Parse streaming mode prevents memory crashes on large CSVs. Current frontend has lib/utils.ts and path aliases configured, but missing react-router-dom/serverless-http and has duplicate job-status routes in demo.ts and jobs.ts.

**Primary recommendation:** Use react-error-boundary library for declarative error handling, TanStack Query polling with status-based termination, incremental TypeScript strict mode migration starting with noImplicitAny, and Papa Parse streaming for CSV preview.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Full-page error state** when components crash (not inline or toast-only)
- **Reload page button** as primary recovery action
- **Technical details** (stack trace, component name) hidden by default, expandable on click
- **Component-specific error boundaries** for high-risk components (Preview, Results) — not just global boundary
- **Preview component error** → show error in preview area only, rest of app continues working
- **Results component error** → show error in results area only, rest of app continues working
- **Dedicated progress page** at /job/:id route (not modal or inline)
- User navigates to progress page after upload completes
- **Step indicator with badges**: (1) Queued → (2) Preprocessing → (3) Training → (4) Scoring
- Current step highlighted, completed steps show checkmarks
- **Progress bar shows BOTH stage percent AND overall job percent**
  - Example: "Training (65% of this stage)" + "Overall: 75% complete"
- **Additional info displayed**:
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FE-01 | React error boundary wraps App component (catches render errors) | react-error-boundary library provides declarative ErrorBoundary component |
| FE-02 | React error boundary wraps high-risk components (Preview, Results) | ErrorBoundary can wrap individual components with custom fallback UI |
| FE-03 | Error boundaries display recovery UI (not blank screen) | Fallback UI with resetErrorBoundary callback enables reload/retry |
| FE-04 | Progress indicator shows multi-stage status (Queued → Preprocessing → Training → Scoring) | shadcn/ui Badge + custom state machine for stage transitions |
| FE-05 | Progress indicator displays percent complete for each stage | Firestore status document includes stageProgress and overallProgress fields |
| FE-06 | Job cancellation UI provides cancel button on job status page | Button triggers DELETE /api/jobs/:id endpoint |
| FE-07 | Job cancellation confirms with user before canceling | shadcn/ui AlertDialog for confirmation modal |
| FE-08 | Polling interval cleanup prevents memory leaks on unmount | TanStack Query's refetchInterval automatically cleans up on unmount |
| FE-09 | Polling stops when job completes (completed/failed/canceled states) | refetchInterval function returns false when status is terminal |
| FE-10 | Polling useEffect dependencies fixed (no stale closures with toast) | TanStack Query eliminates manual useEffect polling entirely |
| FE-11 | Missing dependency lib/utils.ts created (cn utility for shadcn/ui) | **ALREADY EXISTS** - file present with cn function using clsx + tailwind-merge |
| FE-12 | Missing dependency react-router-dom added to package.json | npm install react-router-dom @types/react-router-dom |
| FE-13 | Missing dependency serverless-http added to package.json | npm install serverless-http @types/serverless-http (for Netlify functions) |
| FE-14 | Missing npm script build:client added to package.json | "build:client": "vite build" (client-only build for serverless deploy) |
| FE-15 | Missing npm script dev:server added to package.json | **ALREADY EXISTS** - "dev:server": "tsx server/start.ts" |
| FE-16 | TypeScript strict mode enabled (incremental: noImplicitAny → strictNullChecks → strict) | Incremental migration prevents overwhelming error count |
| FE-17 | TypeScript strict mode violations fixed without type assertions | Use proper types (unknown, null checks) instead of `as any` |
| FE-18 | GCP client instances deduplicated (single Storage/Firestore/PubSub instance) | Export singleton instances from server/config/gcp-clients.ts |
| FE-19 | Duplicate job-status routes resolved (index.ts vs routes/jobs.ts) | Remove demo.ts routes, use only jobs.ts with full security middleware |
| FE-20 | Port mismatch fixed (server uses documented port) | **ALREADY CORRECT** - env.ts defaults to PORT=5001, matches docs |
| FE-21 | CSV parser uses streaming for preview (prevents memory crash on large files) | Papa Parse worker mode + streaming prevents loading entire file into memory |
| FE-22 | File type validation added to click-upload path (not just drag-and-drop) | Move validation to handleFileSelect (shared by both paths) |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| react-error-boundary | 6.1.1 | Declarative error handling | Industry standard (1889+ projects), eliminates need for class components, provides resetErrorBoundary callback |
| @tanstack/react-query | 5.90.12 | Data fetching & polling | **ALREADY INSTALLED** - Declarative refetchInterval, automatic cleanup, built-in status tracking |
| react-router-dom | ^7.2.0 | Client-side routing | De facto standard for React SPAs, needed for /job/:id progress page route |
| papaparse | ^5.5.3 | CSV streaming parser | Fast, handles large files, worker mode prevents UI blocking |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| serverless-http | ^4.0.0 | Express → Serverless adapter | Only if deploying to Netlify Functions (netlify/functions/api.ts exists) |
| typescript-strict-plugin | ^2.4.4 | Incremental strict mode | Optional - enables strict mode per-directory during migration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| react-error-boundary | Class components with componentDidCatch | Library is simpler, eliminates boilerplate, provides hooks (useErrorHandler) |
| TanStack Query polling | Manual useEffect + setInterval | Manual approach error-prone (cleanup, stale closures), Query handles automatically |
| Papa Parse | csv-parse (Node.js) | Papa Parse designed for browser, has worker mode and better large file handling |

**Installation:**
```bash
npm install react-error-boundary react-router-dom papaparse
npm install --save-dev @types/react-router-dom @types/papaparse
# Only if Netlify Functions are used:
npm install serverless-http @types/serverless-http
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/
├── client/
│   ├── components/
│   │   ├── error-boundaries/
│   │   │   ├── RootErrorBoundary.tsx      # App-level boundary (full-page error)
│   │   │   ├── PreviewErrorBoundary.tsx   # Inline error for Preview component
│   │   │   └── ResultsErrorBoundary.tsx   # Inline error for Results component
│   │   ├── progress/
│   │   │   ├── StageIndicator.tsx         # Step badges (Queued → Training)
│   │   │   ├── DualProgressBar.tsx        # Stage % + Overall %
│   │   │   └── JobMetadata.tsx            # Elapsed time, ETA, file info
│   │   └── ui/                            # shadcn/ui components (EXISTING)
│   ├── pages/
│   │   ├── Index.tsx                      # Home/upload page (EXISTING)
│   │   ├── JobProgress.tsx                # NEW: /job/:id progress page
│   │   └── NotFound.tsx                   # 404 page (EXISTING)
│   ├── hooks/
│   │   ├── useJobPolling.ts               # TanStack Query hook for status polling
│   │   └── useJobCancellation.ts          # Hook for cancel confirmation + API call
│   └── utils/
│       ├── csv-parser.ts                  # UPDATE: Add Papa Parse streaming
│       └── api.ts                         # EXISTING API client
└── server/
    ├── config/
    │   ├── gcp-clients.ts                 # NEW: Singleton GCP client instances
    │   └── env.ts                         # EXISTING environment validation
    ├── routes/
    │   ├── jobs.ts                        # EXISTING: Keep this
    │   └── demo.ts                        # DELETE: Duplicate routes
    └── index.ts                           # UPDATE: Remove demo.ts import
```

### Pattern 1: Error Boundary Hierarchy (Granular Recovery)
**What:** Root boundary catches app crashes (full-page error), component boundaries catch localized errors (inline error)
**When to use:** All production apps with async data loading
**Example:**
```tsx
// Root boundary - full-page error
import { ErrorBoundary } from 'react-error-boundary';

function RootFallback({ error, resetErrorBoundary }: FallbackProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <Card className="max-w-lg p-8">
        <h1 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h1>
        <Button onClick={resetErrorBoundary}>Reload Page</Button>
        <Collapsible className="mt-4">
          <CollapsibleTrigger>Show details ▼</CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="text-xs bg-gray-100 p-2 rounded">{error.stack}</pre>
          </CollapsibleContent>
        </Collapsible>
      </Card>
    </div>
  );
}

<ErrorBoundary FallbackComponent={RootFallback}>
  <App />
</ErrorBoundary>

// Component boundary - inline error
function PreviewErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  return (
    <Alert variant="destructive">
      <AlertTitle>Failed to load preview</AlertTitle>
      <AlertDescription>{error.message}</AlertDescription>
      <Button variant="outline" size="sm" onClick={resetErrorBoundary}>Retry</Button>
    </Alert>
  );
}

<ErrorBoundary FallbackComponent={PreviewErrorFallback}>
  <PreviewTable {...props} />
</ErrorBoundary>
```

### Pattern 2: TanStack Query Polling with Conditional Termination
**What:** Use refetchInterval function to stop polling when job reaches terminal state
**When to use:** Long-running async jobs (10-15 min) requiring status updates
**Example:**
```tsx
// Source: TanStack Query v5 docs + GitHub discussions
import { useQuery } from '@tanstack/react-query';

function useJobPolling(jobId: string | null) {
  return useQuery({
    queryKey: ['jobStatus', jobId],
    queryFn: () => checkJobStatus(jobId!),
    enabled: !!jobId, // Don't poll if no jobId
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Terminal states - stop polling
      if (status === 'complete' || status === 'error' || status === 'canceled') {
        return false;
      }
      // Queued/processing - poll every 2 seconds
      return 2000;
    },
    // Automatically stops when component unmounts
  });
}
```

### Pattern 3: Incremental TypeScript Strict Mode Migration
**What:** Enable strict sub-flags one at a time, fixing violations before moving to next flag
**When to use:** Large codebases where enabling `strict: true` generates 100+ errors
**Example:**
```json
// tsconfig.json - Phase 1: Basic type safety
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,  // Start here - forces explicit types
    "strictNullChecks": false,
    // ... other flags false
  }
}

// After fixing all noImplicitAny errors, move to Phase 2:
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": true,  // Enable null/undefined checks
    // ... other flags false
  }
}

// Final phase: Enable all
{
  "compilerOptions": {
    "strict": true  // Enables all strict flags
  }
}
```

### Pattern 4: Papa Parse Streaming for Large CSV Preview
**What:** Use worker mode + step callback to process CSV incrementally
**When to use:** Previewing user-uploaded CSVs (unknown size, could be 100MB+)
**Example:**
```tsx
// Source: Papa Parse docs + react-papaparse examples
import Papa from 'papaparse';

async function parseCSVFile(file: File): Promise<CSVParseResult> {
  return new Promise((resolve, reject) => {
    const rows: any[] = [];

    Papa.parse(file, {
      worker: true,  // Use web worker to avoid blocking UI
      header: true,  // First row is headers
      skipEmptyLines: true,
      preview: 100,  // Only parse first 100 rows for preview
      step: (results) => {
        rows.push(results.data);
      },
      complete: () => {
        resolve({
          rows: rows.slice(0, 20),  // Show first 20
          headers: Object.keys(rows[0] || {}),
          totalRows: rows.length
        });
      },
      error: (error) => reject(error)
    });
  });
}
```

### Pattern 5: Singleton GCP Client Deduplication
**What:** Create GCP clients once, export from central module
**When to use:** Multiple routes/services need Storage/Firestore/PubSub clients
**Example:**
```ts
// server/config/gcp-clients.ts
import { Storage } from '@google-cloud/storage';
import { Firestore } from '@google-cloud/firestore';
import { PubSub } from '@google-cloud/pubsub';
import { env } from './env';

// Create singletons
export const storage = new Storage({ projectId: env.GOOGLE_CLOUD_PROJECT });
export const firestore = new Firestore({ projectId: env.GOOGLE_CLOUD_PROJECT });
export const pubsub = new PubSub({ projectId: env.GOOGLE_CLOUD_PROJECT });

// Usage in routes
import { storage, firestore } from '../config/gcp-clients';
```

### Anti-Patterns to Avoid
- **Manual useEffect polling with setInterval** → Use TanStack Query refetchInterval instead (handles cleanup, stale closures, race conditions)
- **Error boundaries around async event handlers** → Won't catch errors. Use try/catch + toast notifications for async errors
- **Type assertions (`as any`) to pass strict mode** → Defeats purpose of strict mode. Use proper types or `unknown` with type guards
- **Loading entire CSV into memory** → Use streaming (Papa Parse worker mode) to prevent memory crashes
- **Creating new GCP clients in every route** → Singleton pattern prevents connection pool exhaustion

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Error boundaries | Custom class components with componentDidCatch | react-error-boundary | Library handles edge cases (resetKeys, onReset, useErrorHandler hook), tested by 1889+ projects |
| Async polling | useEffect + setInterval + cleanup logic | TanStack Query refetchInterval | Query handles interval cleanup, stale closures, race conditions, enabled flag, automatic stop on unmount |
| CSV parsing | String split + regex for quotes/escapes | Papa Parse | CSV edge cases are hard (quoted commas, newlines in fields, BOM characters), Papa Parse handles all RFC 4180 cases |
| Multi-step progress UI | Custom state machine + CSS | shadcn/ui Badge + Progress components | Pre-built, accessible, consistent with existing UI, Badge has variant system for status colors |
| TypeScript migration | Big-bang strict mode flip | Incremental flag enablement | TypeScript 6.0 assumes strict by default, incremental avoids 1000+ error flood |

**Key insight:** React ecosystem has mature solutions for production UX patterns. Custom implementations rarely handle edge cases as well as battle-tested libraries.

## Common Pitfalls

### Pitfall 1: Error Boundaries Don't Catch Async Errors
**What goes wrong:** Error boundaries only catch errors during React lifecycle (render, lifecycle methods, constructors). Async errors in event handlers, Promises, or setTimeout are NOT caught.
**Why it happens:** JavaScript call stack is different for async code - error doesn't bubble through React render tree.
**How to avoid:** Use try/catch in async handlers, update state to trigger error boundary:
```tsx
// Bad - error boundary won't catch this
async function handleUpload() {
  throw new Error('Upload failed'); // Not caught!
}

// Good - manual error handling
async function handleUpload() {
  try {
    await uploadFile();
  } catch (error) {
    toast({ title: 'Error', description: 'Upload failed' });
  }
}

// Good - trigger error boundary from async code
const [error, setError] = useState<Error | null>(null);
if (error) throw error; // Throws during render, error boundary catches

async function handleUpload() {
  try {
    await uploadFile();
  } catch (error) {
    setError(error); // Next render throws
  }
}
```
**Warning signs:** Unhandled promise rejections in console, no error boundary fallback shown

### Pitfall 2: Polling Continues After Component Unmounts
**What goes wrong:** Manual setInterval polling doesn't stop when user navigates away, causing memory leaks and "Can't perform state update on unmounted component" warnings.
**Why it happens:** setInterval continues running after component unmounts, callback tries to update state.
**How to avoid:** Always clear intervals in useEffect cleanup:
```tsx
// Bad - memory leak
useEffect(() => {
  const interval = setInterval(() => {
    checkStatus(); // setState after unmount!
  }, 2000);
  // Missing cleanup!
}, []);

// Good - cleanup prevents leak
useEffect(() => {
  const interval = setInterval(() => checkStatus(), 2000);
  return () => clearInterval(interval); // Cleanup
}, []);

// Best - TanStack Query handles cleanup automatically
const { data } = useQuery({
  queryKey: ['status'],
  queryFn: checkStatus,
  refetchInterval: 2000
  // Automatically stops on unmount
});
```
**Warning signs:** Console warnings about state updates on unmounted components, intervals running forever

### Pitfall 3: TypeScript Strict Mode Violations Hidden by `any`
**What goes wrong:** Using `as any` or `@ts-ignore` to bypass strict mode errors defeats the purpose of strict mode.
**Why it happens:** Strict mode generates many errors, developers take shortcuts to make code compile.
**How to avoid:** Use proper types. If type is truly unknown, use `unknown` with type guards:
```tsx
// Bad - defeats strict mode
function handleData(data: any) {
  return data.results; // No type safety!
}

// Good - proper typing
interface ApiResponse {
  results: Array<{ id: string; name: string }>;
}
function handleData(data: ApiResponse) {
  return data.results; // Type-safe
}

// Good - unknown with type guard
function handleData(data: unknown) {
  if (isApiResponse(data)) {
    return data.results; // Type-safe after guard
  }
  throw new Error('Invalid data');
}
```
**Warning signs:** Frequent use of `as any`, `@ts-ignore` comments, runtime type errors in production

### Pitfall 4: Loading Large CSV Files into Memory
**What goes wrong:** Using FileReader.readAsText() on 50MB+ CSV files causes browser tab to freeze or crash.
**Why it happens:** FileReader loads entire file contents into JavaScript string, exhausts memory.
**How to avoid:** Use Papa Parse streaming mode with `preview` option:
```tsx
// Bad - loads entire file
reader.readAsText(file);
const content = reader.result as string;
const lines = content.split('\n'); // 100MB in memory!

// Good - streams first 100 rows only
Papa.parse(file, {
  worker: true,
  preview: 100,  // Only parse first 100 rows
  step: (row) => console.log(row)
});
```
**Warning signs:** Browser tab freezes on large file upload, "Out of memory" errors

### Pitfall 5: Duplicate Routes with Different Middleware
**What goes wrong:** Having same route path in multiple routers causes first one to handle request, security middleware on second router is never reached.
**Why it happens:** Express matches first route that fits pattern, doesn't check subsequent routers.
**How to avoid:** Consolidate routes into single router, delete duplicate files:
```tsx
// Bad - demo.ts has unprotected route
router.get('/job-status/:id', async (req, res) => { ... });

// Bad - jobs.ts has protected route (never reached!)
router.get('/job-status/:id', requireAuth, pollLimiter, async (req, res) => { ... });

// Good - single route with all middleware
// In jobs.ts only:
router.get('/job-status/:id', requireAuth, pollLimiter, validateJobId, async (req, res) => { ... });
```
**Warning signs:** Security middleware not executing, rate limiting not working, authentication bypassed

## Code Examples

Verified patterns from official sources:

### TanStack Query Conditional Polling
```tsx
// Source: https://github.com/TanStack/query/discussions/713
// Stops polling when status is terminal
const { data: jobStatus } = useQuery({
  queryKey: ['jobStatus', jobId],
  queryFn: () => fetch(`/api/jobs/${jobId}`).then(r => r.json()),
  refetchInterval: (query) => {
    const status = query.state.data?.status;
    if (['complete', 'error', 'canceled'].includes(status)) {
      return false; // Stop polling
    }
    return 2000; // Poll every 2 seconds
  },
  enabled: !!jobId
});
```

### React Error Boundary with Reset
```tsx
// Source: https://www.npmjs.com/package/react-error-boundary
import { ErrorBoundary } from 'react-error-boundary';

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div role="alert">
      <p>Something went wrong:</p>
      <pre style={{ color: 'red' }}>{error.message}</pre>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  );
}

<ErrorBoundary
  FallbackComponent={ErrorFallback}
  onReset={() => {
    // Reset state that caused error
  }}
  resetKeys={[someKey]} // Auto-reset when key changes
>
  <ComponentThatMayError />
</ErrorBoundary>
```

### Papa Parse Worker Mode
```tsx
// Source: https://www.papaparse.com/
Papa.parse(file, {
  worker: true,        // Use web worker
  header: true,        // First row is headers
  preview: 100,        // Limit to first 100 rows
  skipEmptyLines: true,
  step: (results, parser) => {
    console.log("Row:", results.data);
  },
  complete: (results) => {
    console.log("Parsing complete:", results);
  },
  error: (error) => {
    console.error("Parsing error:", error);
  }
});
```

### shadcn/ui Progress with Multiple Stages
```tsx
// Source: https://ui.shadcn.com/docs/components/progress
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

function StageIndicator({ currentStage, stages }) {
  return (
    <div className="flex items-center gap-2">
      {stages.map((stage, idx) => (
        <Badge
          key={stage}
          variant={
            idx < currentStage ? "default" :  // Completed
            idx === currentStage ? "secondary" :  // Current
            "outline"  // Upcoming
          }
        >
          {idx < currentStage && "✓ "}{stage}
        </Badge>
      ))}
    </div>
  );
}

function DualProgressBar({ stageProgress, overallProgress }) {
  return (
    <div className="space-y-2">
      <div>
        <p className="text-sm text-gray-600">Stage: {stageProgress}%</p>
        <Progress value={stageProgress} className="h-2" />
      </div>
      <div>
        <p className="text-sm text-gray-600">Overall: {overallProgress}%</p>
        <Progress value={overallProgress} className="h-3" />
      </div>
    </div>
  );
}
```

### TypeScript Incremental Strict Mode
```json
// Source: https://oneuptime.com/blog/post/2026-02-20-typescript-strict-mode-guide/view
// Phase 1: Enable noImplicitAny
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": false,
    "strictFunctionTypes": false,
    "strictBindCallApply": false,
    "strictPropertyInitialization": false,
    "noImplicitThis": false,
    "alwaysStrict": false
  }
}

// Phase 2: Enable strictNullChecks
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": true,
    // ... rest false
  }
}

// Phase 3: Enable all (TypeScript 6.0 default)
{
  "compilerOptions": {
    "strict": true
  }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Class component error boundaries | react-error-boundary library | 2021-2022 | Eliminates boilerplate, adds hooks (useErrorHandler), better DX |
| Manual useEffect polling | TanStack Query refetchInterval | 2022-2023 | Automatic cleanup, no stale closures, less code |
| Custom CSV parsers | Papa Parse streaming | 2020-present | Handles RFC 4180 edge cases, worker mode prevents UI blocking |
| TypeScript strict: false default | TypeScript 6.0 assumes strict: true | 2026 | Incremental migration required for existing codebases |
| Multiple GCP client instances | Singleton pattern | Ongoing | Prevents connection pool exhaustion, simpler imports |

**Deprecated/outdated:**
- **FileReader.readAsText for large files**: Causes memory crashes. Use Papa Parse streaming instead.
- **Class components for error boundaries**: react-error-boundary provides better API. Only use classes if can't add dependency.
- **React Query v3 refetchInterval**: v5 improved API with function-based conditional logic (query parameter provides state).

## Validation Architecture

> Nyquist validation is enabled (workflow.nyquist_validation not explicitly false in config.json)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Jest 30.3.0 + React Testing Library |
| Config file | package.json (jest config inline) |
| Quick run command | `npm test -- --testNamePattern="Error boundary\|Polling cleanup"` |
| Full suite command | `npm test` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FE-01 | App-level error boundary catches render errors | unit | `npm test -- frontend/client/__tests__/error-boundaries/RootErrorBoundary.test.tsx` | ❌ Wave 0 |
| FE-02 | Component-specific boundaries isolate errors | unit | `npm test -- frontend/client/__tests__/error-boundaries/PreviewErrorBoundary.test.tsx` | ❌ Wave 0 |
| FE-03 | Error boundary fallback shows recovery UI | unit | `npm test -- frontend/client/__tests__/error-boundaries/ErrorFallback.test.tsx` | ❌ Wave 0 |
| FE-04 | Progress indicator renders stage badges | unit | `npm test -- frontend/client/__tests__/progress/StageIndicator.test.tsx` | ❌ Wave 0 |
| FE-05 | Progress bars show stage and overall percentages | unit | `npm test -- frontend/client/__tests__/progress/DualProgressBar.test.tsx` | ❌ Wave 0 |
| FE-06 | Cancel button appears on job progress page | integration | `npm test -- frontend/client/__tests__/pages/JobProgress.test.tsx` | ❌ Wave 0 |
| FE-07 | Cancel confirmation dialog prevents accidental cancel | integration | `npm test -- frontend/client/__tests__/hooks/useJobCancellation.test.tsx` | ❌ Wave 0 |
| FE-08 | Polling cleanup prevents memory leaks | unit | `npm test -- frontend/client/__tests__/hooks/useJobPolling.test.tsx` | ❌ Wave 0 |
| FE-09 | Polling stops at terminal states | unit | `npm test -- frontend/client/__tests__/hooks/useJobPolling.test.tsx` | ❌ Wave 0 |
| FE-10 | TanStack Query eliminates stale closures | unit | `npm test -- frontend/client/__tests__/hooks/useJobPolling.test.tsx` | ❌ Wave 0 |
| FE-11 | lib/utils.ts exports cn function | smoke | `npm test -- frontend/client/lib/utils.spec.ts` | ✅ EXISTS |
| FE-12 | react-router-dom available for routing | smoke | `npm test -- frontend/client/__tests__/routing.test.tsx` | ❌ Wave 0 |
| FE-13 | serverless-http available (if needed) | smoke | Manual: `npm list serverless-http` | N/A (conditional) |
| FE-14 | build:client script builds frontend only | smoke | Manual: `npm run build:client` | ❌ Wave 0 |
| FE-15 | dev:server script starts Express | smoke | Manual: `npm run dev:server` (already exists) | ✅ EXISTS |
| FE-16 | TypeScript strict flags enabled | smoke | Manual: `npx tsc --noEmit` (check tsconfig) | ❌ Wave 0 |
| FE-17 | No type assertions in strict mode code | smoke | Manual: `grep -r "as any" frontend/client` (should be zero) | ❌ Wave 0 |
| FE-18 | Single GCP client instances exported | unit | `npm test -- frontend/server/__tests__/config/gcp-clients.test.ts` | ❌ Wave 0 |
| FE-19 | No duplicate job-status routes | smoke | Manual: `grep -r "job-status" frontend/server/routes` (jobs.ts only) | ❌ Wave 0 |
| FE-20 | Server listens on PORT=5001 | smoke | Manual: `npm run dev:server` (check output) | ✅ EXISTS |
| FE-21 | CSV parser streams large files | unit | `npm test -- frontend/client/utils/__tests__/csv-parser.test.ts` | ❌ Wave 0 |
| FE-22 | File validation on click-upload path | unit | `npm test -- frontend/client/components/__tests__/Dropzone.test.tsx` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `npm test -- --testNamePattern="<task pattern>" --bail` (run tests for changed components)
- **Per wave merge:** `npm test` (full suite)
- **Phase gate:** Full suite green + manual smoke tests (build:client, dev:server, TypeScript check)

### Wave 0 Gaps
- [ ] `frontend/client/__tests__/error-boundaries/` — covers FE-01, FE-02, FE-03
- [ ] `frontend/client/__tests__/progress/` — covers FE-04, FE-05
- [ ] `frontend/client/__tests__/pages/JobProgress.test.tsx` — covers FE-06
- [ ] `frontend/client/__tests__/hooks/useJobPolling.test.tsx` — covers FE-08, FE-09, FE-10
- [ ] `frontend/client/__tests__/hooks/useJobCancellation.test.tsx` — covers FE-07
- [ ] `frontend/client/__tests__/routing.test.tsx` — covers FE-12
- [ ] `frontend/client/utils/__tests__/csv-parser.test.ts` — covers FE-21
- [ ] `frontend/client/components/__tests__/Dropzone.test.tsx` — covers FE-22
- [ ] `frontend/server/__tests__/config/gcp-clients.test.ts` — covers FE-18
- [ ] Jest config update — add React Testing Library, configure module paths
- [ ] npm scripts: add `build:client` (FE-14)

## Sources

### Primary (HIGH confidence)
- [react-error-boundary npm](https://www.npmjs.com/package/react-error-boundary) - v6.1.1, API, usage patterns
- [TanStack Query useQuery reference](https://tanstack.com/query/v4/docs/framework/react/reference/useQuery) - refetchInterval function API
- [Papa Parse documentation](https://www.papaparse.com/) - streaming mode, worker threads
- [TypeScript Handbook: tsconfig strict](https://www.typescriptlang.org/tsconfig/) - strict flag behavior
- [shadcn/ui Progress component](https://ui.shadcn.com/docs/components/radix/progress) - Progress component API
- [shadcn/ui Badge component](https://ui.shadcn.com/docs/components/radix/badge) - Badge variants for status

### Secondary (MEDIUM confidence)
- [React Error Boundaries – React](https://legacy.reactjs.org/docs/error-boundaries.html) - Official React docs on error boundaries
- [TanStack Query Mastering Polling (Medium, Nov 2025)](https://medium.com/@soodakriti45/tanstack-query-mastering-polling-ee11dc3625cb) - Polling patterns
- [How to Enable TypeScript Strict Mode (OneUpTime, Feb 2026)](https://oneuptime.com/blog/post/2026-02-20-typescript-strict-mode-guide/view) - Incremental migration strategy
- [React useEffect Cleanup Function (Refine)](https://refine.dev/blog/useeffect-cleanup/) - useEffect cleanup patterns
- [Catching Asynchronous Errors in React using Error Boundaries (Medium)](https://medium.com/trabe/catching-asynchronous-errors-in-react-using-error-boundaries-5e8a5fd7b971) - Async error handling workarounds
- [Cancel Asynchronous React App Requests with AbortController (The New Stack)](https://thenewstack.io/cancel-asynchronous-react-app-requests-with-abortcontroller/) - AbortController patterns

### Tertiary (LOW confidence)
- [Express routing best practices (AppMarq)](https://www.appmarq.com/public/efficiency,1020714,Avoid-having-multiple-routes-for-the-same-path-with-Nodejs-Express-App) - Route deduplication patterns
- [Installing shadcn/ui in Vite + React + TypeScript (Medium, Mar 2026)](https://medium.com/@ArcxtheChosen/installing-shadcn-ui-in-vite-react-typescript-complete-guide-03c8cf473d46) - Vite + shadcn setup

## Metadata

**Confidence breakdown:**
- Error boundaries: HIGH - Official React docs + npm library with 1889+ dependents
- Progress indicators: HIGH - shadcn/ui official components + TanStack Query docs
- Polling patterns: HIGH - TanStack Query official docs + GitHub discussions
- TypeScript strict mode: HIGH - Official TypeScript docs + 2026 migration guides
- CSV streaming: HIGH - Papa Parse official docs + npm package (5.5.3)
- Build tooling: HIGH - Existing package.json inspection + Vite official docs

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (30 days - stack is stable, React 18 mature, no breaking changes expected)
