# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework

**Python:**
- Runner: unittest (Python standard library)
- Assertion Library: unittest assertions + numpy/pandas testing utilities
- Config: No pytest.ini or configuration file

**TypeScript:**
- Runner: Vitest
- Assertion Library: Vitest's `expect()` API
- Config: Implicit via Vite

**Run Commands:**
```bash
# Python tests
python -m unittest discover tests/          # Run all tests
python -m unittest tests/model/test_autoencoder.py  # Single file

# Frontend tests (from frontend/ directory)
npm test                                    # Run all tests
```

## Test File Organization

**Python:**
- Location: `tests/` directory tree mirroring source structure
- Pattern: `test_*.py` naming convention
- Structure:
  ```
  tests/
  ├── __init__.py
  ├── dataset/
  │   └── test_loader.py
  ├── features/
  │   └── test_transform.py
  └── model/
      ├── test_autoencoder.py
      └── test_loss.py
  ```

**TypeScript:**
- Location: Co-located with source using `.spec.ts` suffix
- Pattern: `*.spec.ts` naming convention
- Example: `frontend/client/lib/utils.spec.ts`

## Test Structure

**Python Suite Organization:**
```python
import unittest

class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        # Initialize test fixtures
        self.model = AutoencoderModel(cardinalities)
        self.test_data = create_synthetic_df()

    def test_split_train_test(self):
        # Arrange
        df = self.test_data

        # Act
        X_train, X_test = self.model.split_train_test(df, test_size=0.2)

        # Assert
        self.assertEqual(len(X_train) + len(X_test), len(df))
```

**TypeScript Suite Organization:**
```typescript
import { describe, it, expect } from 'vitest';
import { cn } from './utils';

describe('cn utility', () => {
  it('should merge classes correctly', () => {
    expect(cn('foo', 'bar')).toBe('foo bar');
  });

  it('should handle conditional classes', () => {
    expect(cn('foo', false && 'bar')).toBe('foo');
  });
});
```

**Patterns:**
- Python: `setUp()` method for per-test fixtures
- TypeScript: Direct inline setup in each test
- Both use Arrange/Act/Assert pattern

## Mocking

**Python:**
- Limited mocking in current tests
- numpy/pandas testing utilities for array/frame comparison
- `np.testing.assert_array_equal()`, `pd.testing.assert_frame_equal()`

**TypeScript:**
- No mocking observed in current tests
- Pure function testing only

**What to Mock:**
- External API calls (GCS, Firestore, Pub/Sub)
- File system operations
- Network requests

**What NOT to Mock:**
- Pure functions
- Data transformations
- Internal utilities

## Fixtures and Factories

**Python Test Data:**
```python
def setUp(self):
    # Synthetic DataFrame creation
    self.test_df = pd.DataFrame({
        'col1': ['A', 'B', 'C'],
        'col2': [1, 2, 3]
    })

    # Model initialization with test cardinalities
    self.model = AutoencoderModel([3, 2])
```

**Location:**
- Python: Inline in test files, created in `setUp()`
- TypeScript: Inline in test blocks

## Coverage

**Requirements:**
- No enforced coverage target
- No coverage tooling configured

**Gaps:**
- No tests for `main.py` CLI commands
- No tests for `worker.py`
- No tests for `utils.py`
- No integration tests for full pipeline
- No E2E tests for web application

## Test Types

**Unit Tests:**
- Scope: Single class/function in isolation
- Location: `tests/model/`, `tests/features/`, `tests/dataset/`
- Examples:
  - `test_autoencoder.py` - Model split/train methods
  - `test_loss.py` - Loss function calculations
  - `test_transform.py` - Vectorization/de-vectorization

**Integration/Manual Tests:**
- Location: Root directory
- Files:
  - `test_local_upload.py` - Manual CSV upload testing
  - `test_worker_local.py` - Local worker testing
- Purpose: End-to-end workflow validation (run manually)

**E2E Tests:**
- Not currently implemented

## Common Patterns

**Async Testing (Python):**
```python
# Not currently used - tests are synchronous
```

**Error Testing (Python):**
```python
def test_invalid_input(self):
    with self.assertRaises(ValueError):
        self.model.process(invalid_data)
```

**DataFrame Comparison:**
```python
pd.testing.assert_frame_equal(result_df, expected_df)
```

**Array Comparison:**
```python
np.testing.assert_array_equal(result_array, expected_array)
```

**Snapshot Testing:**
- Not used in this codebase

## Test Files Reference

| File Path | Purpose |
|-----------|---------|
| `tests/model/test_autoencoder.py` | Autoencoder model unit tests |
| `tests/model/test_loss.py` | Custom loss function tests |
| `tests/features/test_transform.py` | Table2Vector transformation tests |
| `tests/dataset/test_loader.py` | DataLoader unit tests |
| `frontend/client/lib/utils.spec.ts` | Frontend utility function tests |
| `test_local_upload.py` | Manual upload testing script |
| `test_worker_local.py` | Manual worker testing script |

## Notable Observations

1. **Asymmetric Coverage**: Backend ML code has tests; frontend has minimal coverage
2. **No CI Integration**: Tests must be run manually
3. **Missing CLI Tests**: No tests for main.py commands
4. **No Mocking Framework**: Tests don't mock external services
5. **Manual Integration Tests**: Separate scripts for end-to-end testing

---

*Testing analysis: 2026-01-23*
*Update when test patterns change*
