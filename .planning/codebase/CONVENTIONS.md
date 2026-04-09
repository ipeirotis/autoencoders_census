# Coding Conventions

**Analysis Date:** 2026-01-23

## Naming Patterns

**Files:**
- Python modules: snake_case (`autoencoder.py`, `loader.py`, `trainer.py`)
- TypeScript components: PascalCase (`ResultCard.tsx`, `NaNCheckbox.tsx`)
- UI components: kebab-case (`card.tsx`, `button.tsx`, `use-toast.ts`)
- Test files: `test_*.py` (Python), `*.spec.ts` (TypeScript)

**Functions:**
- Python: snake_case (`load_2015()`, `vectorize_table()`, `set_seed()`)
- TypeScript: camelCase (`parseCSVFile()`, `checkJobStatus()`, `getUploadUrl()`)
- React handlers: camelCase (`handleClick`, `onSubmit`)

**Variables:**
- Python: snake_case (`variable_types`, `attribute_cardinalities`)
- TypeScript: camelCase (`jobId`, `bucketName`, `filePath`)
- Constants: UPPER_SNAKE_CASE (`INPUT_SHAPE`, `SEP`, `MISSING`, `API_BASE`)

**Types:**
- Python classes: PascalCase (`AutoencoderModel`, `DataLoader`, `Table2Vector`)
- TypeScript interfaces: PascalCase with `Props` suffix (`ResultCardProps`, `NaNCheckboxProps`)
- TypeScript types: PascalCase (`CSVParseResult`, `JobStatus`)

## Code Style

**Formatting:**
- Frontend: Prettier with `.prettierrc`
  - Tab width: 2 spaces
  - Trailing commas: "all"
  - No tabs (use spaces)
- Python: Implicit PEP 8 (no linter configured)
  - 4-space indentation
  - No explicit line length limit

**Linting:**
- TypeScript: No ESLint configured
- TypeScript strict mode: Disabled (`strict: false` in tsconfig.json)
- Python: No linting tools (no pylint, flake8, black)

## Import Organization

**Python:**
1. Standard library imports
2. Third-party imports (tensorflow, pandas, numpy)
3. Local imports (model, dataset, features)

**TypeScript:**
1. External packages (react, express)
2. Internal modules (@/lib, @/components)
3. Relative imports (./utils, ../types)
4. Type imports (import type {})

**Path Aliases (TypeScript):**
- `@/*` → `./client/*`
- `@shared/*` → `./shared/*`

## Error Handling

**Patterns:**
- Python: try/except at CLI command boundaries
- TypeScript: Promise rejection handling, React error states
- Note: Some bare `except:` clauses exist (should use specific exceptions)

**Error Types:**
- Python: Raises standard exceptions, caught at entry points
- TypeScript: Throws errors, uses toast notifications for user feedback

**Logging:**
- Python: `logging` module with StreamHandler to stdout
- TypeScript: `console.log` for development
- Note: Debug print statements mixed with logging (should consolidate)

## Logging

**Framework:**
- Python: `logging.Logger(__name__)` (incorrectly initialized in some files)
- TypeScript: Console methods

**Patterns:**
- Log level: DEBUG in development
- Output: stdout/console
- Note: Should use `logging.getLogger(__name__)` not `logging.Logger(__name__)`

## Comments

**When to Comment:**
- Complex algorithms (loss calculations, data transformations)
- Business logic (column mappings, preprocessing steps)
- Section separation in large files

**Docstrings (Python):**
- Module docstrings at file top (triple-quoted)
- Class docstrings after definition
- Method docstrings for public APIs
- Format: Descriptive with Args/Returns sections

**JSDoc (TypeScript):**
- Used sparingly for complex functions
- Inline comments (`//`) for explanatory notes

**TODO Comments:**
- Format: `# TODO:` or `// TODO:`
- Found in: `main.py`, `loss.py` (commented-out alternatives)

## Function Design

**Size:**
- Large functions exist (some 100+ lines)
- Should be refactored into smaller units

**Parameters:**
- Python: Positional and keyword arguments
- TypeScript: Destructured objects for props

**Return Values:**
- Python: Single values, tuples, or DataFrames
- TypeScript: Typed return values, Promises for async

## Module Design

**Exports:**
- Python: Classes and functions at module level
- TypeScript: Named exports preferred
- React: Default export for components

**Barrel Files:**
- TypeScript: `index.ts` for Express server
- Python: `__init__.py` for packages

## Configuration Patterns

**Python:**
- YAML files for hyperparameters (`config/*.yaml`)
- Environment variables via `python-dotenv`
- CLI arguments via `click`

**TypeScript:**
- Environment variables via `.env`
- TypeScript config via `tsconfig.json`
- Vite config via `vite.config.ts`

---

*Convention analysis: 2026-01-23*
*Update when patterns change*
