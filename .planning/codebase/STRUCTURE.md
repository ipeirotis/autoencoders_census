# Codebase Structure

**Analysis Date:** 2026-01-23

## Directory Layout

```
AutoEncoder2025/
├── config/             # ML hyperparameter configurations
├── dataset/            # Data loading module
├── evaluate/           # Evaluation and outlier detection
├── features/           # Feature transformation
├── frontend/           # React + Express web application
│   ├── client/        # React UI
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── pages/
│   │   └── utils/
│   ├── netlify/       # Serverless functions
│   ├── server/        # Express API
│   │   └── routes/
│   └── shared/        # Shared types
├── model/              # Neural network models
├── tests/              # Python test suite
│   ├── dataset/
│   ├── features/
│   └── model/
├── train/              # Training orchestration
├── main.py             # CLI entry point
├── worker.py           # Pub/Sub worker
├── utils.py            # Shared utilities
├── Dockerfile          # Vertex AI container
└── requirements.txt    # Python dependencies
```

## Directory Purposes

**config/**
- Purpose: YAML hyperparameter configurations
- Contains: `simple_autoencoder.yaml`, `hp_autoencoder.yaml`, VAE configs
- Key files: Training parameters, tuning search spaces

**dataset/**
- Purpose: Data loading and preprocessing
- Contains: `loader.py` - DataLoader class with multi-dataset support
- Key files: `loader.py` handles SADC, Pennycook, bot detection datasets

**evaluate/**
- Purpose: Model evaluation and outlier detection
- Contains: `evaluator.py`, `generator.py`, `outliers.py`
- Key files: Reconstruction error computation, sample generation

**features/**
- Purpose: Feature transformation for tabular data
- Contains: `transform.py` - Table2Vector class
- Key files: One-hot encoding, categorical vectorization

**frontend/**
- Purpose: Web application (React + Express)
- Contains: Client UI, server API, shared types
- Subdirectories:
  - `client/` - React components, hooks, pages
  - `server/` - Express routes and middleware
  - `shared/` - TypeScript type definitions
  - `netlify/` - Serverless function wrappers

**frontend/client/**
- Purpose: React UI application
- Contains: Components, pages, utilities
- Key files:
  - `App.tsx` - Main entry point
  - `pages/Index.tsx` - Primary UI page
  - `components/Dropzone.tsx` - File upload
  - `components/PreviewTable.tsx` - CSV preview and results

**frontend/server/**
- Purpose: Express API server
- Contains: Route handlers, middleware, GCP client init
- Key files:
  - `index.ts` - Server setup, Multer, GCP clients
  - `routes/jobs.ts` - Job management endpoints

**model/**
- Purpose: Neural network architecture definitions
- Contains: Model classes, custom layers, loss functions
- Key files:
  - `autoencoder.py` - AutoencoderModel class
  - `variational_autoencoder.py` - VAE implementation
  - `loss.py` - CustomCategoricalCrossentropyAE
  - `factory.py` - Model instantiation

**tests/**
- Purpose: Python unit tests
- Contains: Tests organized by module
- Subdirectories: `dataset/`, `features/`, `model/`

**train/**
- Purpose: Training orchestration
- Contains: `trainer.py`, `task.py`
- Key files:
  - `trainer.py` - Trainer class with train/search methods
  - `task.py` - Vertex AI container entry point

## Key File Locations

**Entry Points:**
- `main.py` - CLI entry (train, evaluate, find_outliers, generate)
- `worker.py` - Pub/Sub worker for Vertex AI dispatch
- `frontend/server/index.ts` - Express server startup
- `frontend/client/App.tsx` - React app entry
- `train/task.py` - Vertex AI container entry

**Configuration:**
- `config/*.yaml` - ML hyperparameters
- `frontend/tsconfig.json` - TypeScript config
- `frontend/vite.config.ts` - Vite bundler config
- `frontend/tailwind.config.ts` - Tailwind CSS config
- `.env` / `.env.example` - Environment variables

**Core Logic:**
- `model/autoencoder.py` - Autoencoder architecture
- `model/loss.py` - Custom loss function
- `dataset/loader.py` - Data loading (20K+ lines)
- `features/transform.py` - Categorical encoding
- `train/trainer.py` - Training orchestration
- `evaluate/evaluator.py` - Reconstruction metrics

**Testing:**
- `tests/model/test_autoencoder.py` - Model tests
- `tests/features/test_transform.py` - Transform tests
- `frontend/client/lib/utils.spec.ts` - Frontend tests

**Documentation:**
- `README.md` - Project overview

## Naming Conventions

**Files:**
- Python modules: snake_case (`autoencoder.py`, `loader.py`)
- TypeScript components: PascalCase (`ResultCard.tsx`, `Dropzone.tsx`)
- UI components: kebab-case (`card.tsx`, `button.tsx`)
- Test files: `test_*.py` (Python), `*.spec.ts` (TypeScript)

**Directories:**
- Lowercase, singular/plural descriptive: `model`, `dataset`, `features`
- Frontend: `client/`, `server/`, `shared/`
- Feature grouping: `components/`, `hooks/`, `utils/`

**Special Patterns:**
- `__init__.py` for Python packages
- `index.ts` for Express server entry
- `App.tsx` for React app entry

## Where to Add New Code

**New ML Model:**
- Implementation: `model/{model_name}.py`
- Factory registration: `model/factory.py`
- Tests: `tests/model/test_{model_name}.py`

**New Dataset Support:**
- Implementation: `dataset/loader.py` (add load method)
- Column mappings: `utils.py` → `define_necessary_elements()`

**New API Endpoint:**
- Route handler: `frontend/server/routes/{feature}.ts`
- Registration: `frontend/server/index.ts`

**New React Component:**
- Component: `frontend/client/components/{Name}.tsx`
- UI primitives: `frontend/client/components/ui/`

**New CLI Command:**
- Implementation: `main.py` (add @click.command)

## Special Directories

**cache/**
- Purpose: Trained model outputs
- Source: Generated during training
- Committed: No (should be gitignored)

**node_modules/**
- Purpose: npm dependencies
- Source: npm install
- Committed: No

**data/**
- Purpose: Training datasets
- Source: External data files
- Committed: No (gitignored)

---

*Structure analysis: 2026-01-23*
*Update when directory structure changes*
