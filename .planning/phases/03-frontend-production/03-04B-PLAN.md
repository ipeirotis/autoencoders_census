---
phase: 03-frontend-production
plan: 04B
type: execute
wave: 1
depends_on: []
files_modified:
  - frontend/client/utils/csv-parser.ts
  - frontend/client/components/Dropzone.tsx
autonomous: true
requirements: [FE-21, FE-22]

must_haves:
  truths:
    - "CSV parser uses streaming to prevent memory crashes on large files"
    - "File type validation applied to both drag-drop and click-upload paths"
    - "Binary files disguised as CSV are rejected"
  artifacts:
    - path: "frontend/client/utils/csv-parser.ts"
      provides: "Papa Parse streaming parser"
      contains: "Papa.parse"
      min_lines: 30
  key_links:
    - from: "frontend/client/components/Dropzone.tsx"
      to: "csv-parser.ts streaming"
      via: "import and function call"
      pattern: "import.*parseCSVFile"
---

<objective>
Fix memory and security gaps in CSV file handling.

Purpose: Prevent memory crashes from large CSV files (50MB+) and ensure file validation covers all upload paths. Close security gap where click-upload bypasses validation.
Output: Streaming CSV parser with Papa Parse, unified file validation for both upload paths.
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

# Current gaps (from CONTEXT.md)
- FE-21: CSV parser memory issue - Non-streaming preview crashes on large files
- FE-22: File type validation gap - Click-upload path lacks validation (only drag-drop has it)
</context>

<interfaces>
<!-- Papa Parse API -->
From papaparse library (installed in 03-01):
```typescript
import Papa from 'papaparse';

interface ParseResult<T> {
  data: T[];
  errors: Array<{ message: string; row?: number }>;
  meta: {
    delimiter: string;
    linebreak: string;
    aborted: boolean;
    truncated: boolean;
    cursor: number;
  };
}

Papa.parse(file: File, config: {
  worker?: boolean;        // Use web worker
  header?: boolean;        // First row is headers
  preview?: number;        // Limit rows parsed
  skipEmptyLines?: boolean;
  step?: (results: ParseResult<any>, parser: Papa.Parser) => void;
  complete?: (results: ParseResult<any>) => void;
  error?: (error: Error) => void;
}): void;
```

Existing file-type library (from Phase 01):
```typescript
import { fileTypeFromBuffer } from 'file-type';

// Returns file type from magic bytes
const type = await fileTypeFromBuffer(buffer);
// type.mime === 'text/csv' or other MIME type
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create streaming CSV parser with Papa Parse</name>
  <files>frontend/client/utils/csv-parser.ts</files>
  <action>
Create streaming CSV parser to prevent memory crashes on large files (FE-21).

Pattern (from research Pattern 4):
```typescript
import Papa from 'papaparse';

export interface CSVParseResult {
  rows: Array<Record<string, any>>;
  headers: string[];
  totalRows: number;
}

export async function parseCSVFile(file: File): Promise<CSVParseResult> {
  return new Promise((resolve, reject) => {
    const rows: Array<Record<string, any>> = [];

    Papa.parse(file, {
      worker: true,        // Use web worker to avoid blocking UI
      header: true,        // First row is headers
      skipEmptyLines: true,
      preview: 100,        // Only parse first 100 rows for preview (prevents memory crash)
      step: (results) => {
        rows.push(results.data);
      },
      complete: (results) => {
        resolve({
          rows: rows.slice(0, 20),  // Show first 20 rows in preview
          headers: Object.keys(rows[0] || {}),
          totalRows: rows.length
        });
      },
      error: (error) => {
        reject(new Error(`CSV parsing failed: ${error.message}`));
      }
    });
  });
}
```

Worker mode prevents UI blocking. Preview limit (100 rows) prevents loading entire 50MB+ file into memory. Only return first 20 rows for display.

Update PreviewTable.tsx to use this new parseCSVFile function instead of FileReader.readAsText (if used).
  </action>
  <verify>
    <automated>grep -q "Papa.parse" frontend/client/utils/csv-parser.ts && grep -q "worker: true" frontend/client/utils/csv-parser.ts && grep -q "preview:" frontend/client/utils/csv-parser.ts && echo "SUCCESS"</automated>
  </verify>
  <done>csv-parser.ts created with Papa Parse streaming. worker: true prevents UI blocking. preview: 100 limits memory usage. Only 20 rows returned for display. Error handling wraps Papa Parse errors.</done>
</task>

<task type="auto">
  <name>Task 2: Add file validation to click-upload path</name>
  <files>frontend/client/components/Dropzone.tsx</files>
  <action>
Research found file type validation only on drag-drop path, not click-upload path (FE-22).

Move validation to shared function called by both paths:

```typescript
import { fileTypeFromBuffer } from 'file-type';

async function validateFile(file: File): Promise<{ valid: boolean; error?: string }> {
  // Check extension
  if (!file.name.endsWith('.csv')) {
    return { valid: false, error: 'Only CSV files are allowed' };
  }

  // Check magic bytes (prevents .exe renamed to .csv)
  const buffer = await file.arrayBuffer();
  const type = await fileTypeFromBuffer(new Uint8Array(buffer));

  // CSV files are text/plain or text/csv, may not have magic bytes
  // If fileTypeFromBuffer returns undefined (text file), that's OK for CSV
  // If it returns a type, ensure it's NOT a binary format
  if (type && !['text/csv', 'text/plain'].includes(type.mime)) {
    return { valid: false, error: `Invalid file type: ${type.mime}. Only CSV files allowed.` };
  }

  return { valid: true };
}

// Call from both paths
const handleDrop = async (acceptedFiles: File[]) => {
  const file = acceptedFiles[0];
  const validation = await validateFile(file);
  if (!validation.valid) {
    toast({ title: 'Error', description: validation.error });
    return;
  }
  // ... continue with upload
};

const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const validation = await validateFile(file);
  if (!validation.valid) {
    toast({ title: 'Error', description: validation.error });
    return;
  }
  // ... continue with upload
};
```

Pattern from Phase 01: Use file-type library to check magic bytes, not just extensions.
  </action>
  <verify>
    <automated>grep -q "validateFile" frontend/client/components/Dropzone.tsx && grep -c "validateFile" frontend/client/components/Dropzone.tsx | grep -q "^3$" && echo "SUCCESS"</automated>
  </verify>
  <done>Dropzone.tsx updated with shared validateFile function. Both drag-drop and click-upload paths call validateFile. Magic bytes checked with file-type library. Extension and binary format validation applied to both upload methods.</done>
</task>

</tasks>

<verification>
Run all automated task verifications:
1. csv-parser.ts uses Papa Parse with worker mode and preview limit
2. Dropzone.tsx validateFile called from both upload paths (3 occurrences: definition + 2 calls)

Manual testing:
1. Upload large CSV (10MB+) via drag-drop → should not freeze browser
2. Upload large CSV (10MB+) via click button → should not freeze browser
3. Upload .exe renamed to .csv via drag-drop → should be rejected
4. Upload .exe renamed to .csv via click button → should be rejected
5. Upload valid CSV via drag-drop → should work
6. Upload valid CSV via click button → should work

Performance test:
```bash
# Create large test CSV
seq 1 100000 | awk '{print $1",value"$1",data"$1}' > large-test.csv
# Upload via UI - should show first 20 rows only, not crash
```
</verification>

<success_criteria>
- [x] csv-parser.ts created with Papa Parse streaming
- [x] Papa Parse uses worker mode and preview limit
- [x] Dropzone.tsx validateFile shared by both upload paths
- [x] Magic bytes checked for binary file detection
- [x] Large CSV upload doesn't crash browser (streaming)
- [x] Binary files (exe, zip) renamed to .csv are rejected
- [x] Both drag-drop and click-upload validated identically
</success_criteria>

<output>
After completion, create `.planning/phases/03-frontend-production/03-04B-SUMMARY.md`
</output>
