---
phase: 03-frontend-production
plan: 04A
type: execute
wave: 1
depends_on: []
files_modified:
  - frontend/tsconfig.json
  - frontend/client/App.tsx
  - frontend/client/pages/Index.tsx
  - frontend/client/components/Dropzone.tsx
  - frontend/client/components/PreviewTable.tsx
  - frontend/client/components/ResultCard.tsx
  - frontend/client/components/NaNCheckbox.tsx
  - frontend/server/index.ts
  - frontend/server/routes/auth.ts
  - frontend/server/routes/jobs.ts
autonomous: true
requirements: [FE-16, FE-17]

must_haves:
  truths:
    - "TypeScript strict mode enabled with noImplicitAny as first step"
    - "No type assertions (as any) used to bypass strict mode checks"
    - "All client and server code passes TypeScript compilation"
  artifacts:
    - path: "frontend/tsconfig.json"
      provides: "noImplicitAny enabled (first strict flag)"
      contains: '"noImplicitAny": true'
  key_links:
    - from: "all TypeScript files"
      to: "tsconfig.json"
      via: "TypeScript compiler"
      pattern: "noImplicitAny"
---

<objective>
Enable TypeScript strict mode incrementally starting with noImplicitAny.

Purpose: Improve type safety without overwhelming error count. First step of three-phase strict mode migration (noImplicitAny → strictNullChecks → strict: true).
Output: noImplicitAny enabled, all type errors fixed with proper types (no `as any` assertions).
</objective>

<execution_context>
@/Users/aaron/.claude/get-shit-done/workflows/execute-plan.md
@/Users/aaron/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/03-frontend-production/03-CONTEXT.md
@.planning/phases/03-frontend-production/03-RESEARCH.md

# Incremental strict mode strategy (from research Pattern 3)
Phase 1 (this plan): Enable noImplicitAny (forces explicit types)
Phase 2 (future): Enable strictNullChecks (after Phase 1 violations fixed)
Phase 3 (future): Enable strict: true (after Phase 2 violations fixed)

User decision: Incremental vs all-at-once is Claude's discretion. Research recommends incremental to avoid 1000+ error flood.
</context>

<interfaces>
<!-- React and Express types -->
React event types:
```typescript
import React from 'react';

type ChangeEvent = React.ChangeEvent<HTMLInputElement>;
type FormEvent = React.FormEvent<HTMLFormElement>;
type MouseEvent = React.MouseEvent<HTMLButtonElement>;
```

Express types:
```typescript
import { Request, Response, NextFunction } from 'express';

// Basic handler
(req: Request, res: Response) => void

// With middleware
(req: Request, res: Response, next: NextFunction) => void

// With typed params
interface JobParams {
  id: string;
}
router.get('/jobs/:id', (req: Request<JobParams>, res: Response) => {
  const id = req.params.id; // string, not any
});
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Enable noImplicitAny in tsconfig.json</name>
  <files>frontend/tsconfig.json</files>
  <action>
Enable first strict mode flag following incremental migration strategy (FE-16).

Update tsconfig.json:
```json
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,  // NEW: Force explicit types
    "strictNullChecks": false,  // Defer to next phase
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": false
    // ... other options unchanged
  }
}
```

Run `npx tsc --noEmit` to see how many errors are generated. If count is manageable (< 50), proceed to fix. If > 50, consider breaking into sub-plans.

Per research Pattern 3: noImplicitAny is first step, catches parameters and variables missing type annotations.
  </action>
  <verify>
    <automated>grep -q '"noImplicitAny": true' frontend/tsconfig.json && echo "SUCCESS"</automated>
  </verify>
  <done>tsconfig.json updated with noImplicitAny: true. Other strict flags remain false for incremental migration.</done>
</task>

<task type="auto">
  <name>Task 2: Fix noImplicitAny violations in client code</name>
  <files>frontend/client/App.tsx, frontend/client/pages/Index.tsx, frontend/client/components/Dropzone.tsx, frontend/client/components/PreviewTable.tsx, frontend/client/components/ResultCard.tsx, frontend/client/components/NaNCheckbox.tsx</files>
  <action>
Fix implicit any type errors revealed by noImplicitAny (FE-17).

Common patterns to fix:
1. Event handlers: `(e) =>` becomes `(e: React.ChangeEvent<HTMLInputElement>) =>`
2. Callback parameters: `(item) =>` becomes `(item: ItemType) =>`
3. useState with no initial value: `useState()` becomes `useState<Type | null>(null)`
4. Props destructuring: `({ prop })` becomes explicit interface

DO NOT use `as any` type assertions. Use proper types (FE-17).

Example fixes:
```typescript
// Before (implicit any)
const handleChange = (e) => setValue(e.target.value);

// After (explicit type)
const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => setValue(e.target.value);

// Before (implicit any)
const [data, setData] = useState();

// After (explicit type)
const [data, setData] = useState<DataType | null>(null);
```

If type is truly unknown, use `unknown` with type guards instead of `any`.

Run `npx tsc --noEmit` after fixes to verify all noImplicitAny errors resolved.
  </action>
  <verify>
    <automated>cd frontend && npx tsc --noEmit 2>&1 | grep -q "Found 0 errors" && echo "SUCCESS"</automated>
  </verify>
  <done>All noImplicitAny violations fixed in client code. Event handlers typed. useState calls typed. No `as any` assertions used. TypeScript compilation succeeds with 0 errors.</done>
</task>

<task type="auto">
  <name>Task 3: Fix noImplicitAny violations in server code</name>
  <files>frontend/server/index.ts, frontend/server/routes/auth.ts, frontend/server/routes/jobs.ts</files>
  <action>
Fix implicit any type errors in Express server code (FE-17).

Common patterns in Express:
1. Request handlers: `(req, res)` becomes `(req: Request, res: Response)`
2. Middleware: `(req, res, next)` becomes `(req: Request, res: Response, next: NextFunction)`
3. Error handlers: `(err, req, res, next)` becomes `(err: Error, req: Request, res: Response, next: NextFunction)`
4. Route parameters: Use `req.params as { id: string }` or create interface

Example:
```typescript
import { Request, Response, NextFunction } from 'express';

// Before
router.get('/jobs/:id', (req, res) => {
  const id = req.params.id;
});

// After
router.get('/jobs/:id', (req: Request, res: Response) => {
  const id = req.params.id as string;
});

// Or with interface
interface JobParams {
  id: string;
}

router.get('/jobs/:id', (req: Request<JobParams>, res: Response) => {
  const id = req.params.id;
});
```

DO NOT use `as any`. Use Express types from @types/express.

Run `npx tsc --noEmit` after fixes.
  </action>
  <verify>
    <automated>cd frontend && npx tsc --noEmit 2>&1 | grep -q "Found 0 errors" && echo "SUCCESS"</automated>
  </verify>
  <done>All noImplicitAny violations fixed in server code. Express handlers typed with Request/Response. Middleware typed with NextFunction. No `as any` assertions. TypeScript compilation succeeds.</done>
</task>

</tasks>

<verification>
Run all automated task verifications:
1. tsconfig.json has noImplicitAny: true
2. Client code TypeScript compilation succeeds (0 errors)
3. Server code TypeScript compilation succeeds (0 errors)

Full build verification:
```bash
cd frontend
npx tsc --noEmit  # Should show 0 errors
npm run build:client  # Should complete successfully
npm test  # All tests should pass
```

Verify no `as any` used:
```bash
grep -r "as any" frontend/client/ frontend/server/ | grep -v node_modules | wc -l
# Should be 0
```
</verification>

<success_criteria>
- [x] noImplicitAny enabled in tsconfig.json
- [x] All client code noImplicitAny violations fixed
- [x] All server code noImplicitAny violations fixed
- [x] No `as any` type assertions used (proper types instead)
- [x] TypeScript compilation succeeds with 0 errors
- [x] npm test passes (no regressions)
</success_criteria>

<output>
After completion, create `.planning/phases/03-frontend-production/03-04A-SUMMARY.md`

Note for future phases:
- Phase 2 of strict mode: Enable strictNullChecks after noImplicitAny issues resolved
- Phase 3 of strict mode: Enable strict: true after strictNullChecks issues resolved
</output>
