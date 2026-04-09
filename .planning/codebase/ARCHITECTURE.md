# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Hybrid Full-Stack Application (Python ML + Node.js API + React UI)

**Key Characteristics:**
- Microservices with asynchronous job queue
- Distributed cloud-native architecture
- Clear separation between ML pipeline and web tier
- Event-driven processing via Pub/Sub

## Layers

**Presentation Layer:**
- Purpose: User interface for data upload and results display
- Contains: React components, file upload, CSV preview, results visualization
- Location: `frontend/client/`
- Depends on: API Gateway layer
- Used by: End users via browser

**API Gateway Layer:**
- Purpose: HTTP API for file orchestration and job management
- Contains: Express routes, Google Cloud client initialization, Multer middleware
- Location: `frontend/server/index.ts`, `frontend/server/routes/jobs.ts`
- Depends on: Message Queue layer, Data Persistence layer
- Used by: Presentation layer

**Message Queue Layer:**
- Purpose: Asynchronous job dispatch and decoupling
- Contains: Pub/Sub topic publishing and subscription handling
- Location: `frontend/server/routes/jobs.ts` (publish), `worker.py` (subscribe)
- Depends on: Worker/Orchestrator layer
- Used by: API Gateway layer

**Worker/Orchestrator Layer:**
- Purpose: Background job dispatch to Vertex AI
- Contains: Pub/Sub listener, Vertex AI job triggers, job monitoring
- Location: `worker.py`
- Depends on: ML Pipeline layer
- Used by: Message Queue layer

**ML Pipeline Layer (Containerized):**
- Purpose: Data processing, model training, evaluation
- Contains: DataLoader, Table2Vector, Trainer, Evaluator
- Location: `dataset/`, `features/`, `train/`, `evaluate/`
- Depends on: Model layer, Data Persistence layer
- Used by: Worker/Orchestrator layer

**Model Layer:**
- Purpose: Neural network architecture definitions
- Contains: AutoencoderModel, VariationalAutoencoderModel, LinearAutoencoder, custom losses
- Location: `model/`
- Depends on: TensorFlow/Keras
- Used by: ML Pipeline layer

**Data Persistence Layer:**
- Purpose: Storage for files and metadata
- Contains: GCS file operations, Firestore document operations
- Location: Distributed across layers
- Depends on: Google Cloud SDK
- Used by: All layers

## Data Flow

**Upload & Training Flow:**

1. User selects CSV → Browser
2. Frontend requests signed URL from `/api/jobs/upload-url`
3. Browser uploads CSV directly to GCS via signed URL
4. Frontend calls POST `/api/jobs/start-job`
5. Express creates Firestore doc with `status: "uploading"`
6. Express publishes Pub/Sub message to `job-upload-topic`
7. `worker.py` receives message via subscription
8. Worker triggers Vertex AI CustomContainerTrainingJob
9. Vertex AI container downloads CSV from GCS
10. Container runs training/outlier detection pipeline
11. Container writes results to Firestore
12. Frontend polls GET `/api/jobs/job-status/:id` every 2 seconds
13. Frontend displays results when `status == "complete"`

**State Management:**
- File-based: Training data in GCS
- Document-based: Job metadata in Firestore
- Stateless: Each HTTP request independent

## Key Abstractions

**Model Factory:**
- Purpose: Instantiate model variants (AE/VAE/PCA)
- Examples: `get_model(model_name, cardinalities)` in `model/factory.py`
- Pattern: Factory method

**DataLoader:**
- Purpose: Pluggable data loading with dataset-specific preprocessing
- Examples: `load_2015()`, `load_2017()`, `load_pennycook_1()` in `dataset/loader.py`
- Pattern: Strategy pattern for different datasets

**Table2Vector:**
- Purpose: Categorical encoding for tabular data
- Examples: `vectorize_table()`, `tabularize_vector()` in `features/transform.py`
- Pattern: Transformer/Encoder

**Trainer:**
- Purpose: Model training orchestration
- Examples: `train()`, `search_hyperparameters()` in `train/trainer.py`
- Pattern: Template method

**Evaluator:**
- Purpose: Reconstruction error computation
- Examples: `impute_missing_values()`, `predict()` in `evaluate/evaluator.py`
- Pattern: Strategy pattern

## Entry Points

**CLI Entry:**
- Location: `main.py`
- Triggers: User runs `python main.py <command>`
- Responsibilities: Parse CLI args, execute training/evaluation commands

**Worker Entry:**
- Location: `worker.py`
- Triggers: Pub/Sub message received
- Responsibilities: Trigger Vertex AI jobs, monitor completion

**API Server Entry:**
- Location: `frontend/server/index.ts`
- Triggers: HTTP requests
- Responsibilities: Route requests, manage GCS/Firestore/Pub/Sub clients

**Vertex AI Container Entry:**
- Location: `train/task.py`
- Triggers: Vertex AI job start
- Responsibilities: Download data, run training, save results

**React App Entry:**
- Location: `frontend/client/App.tsx`
- Triggers: Browser page load
- Responsibilities: Render UI, handle user interactions

## Error Handling

**Strategy:** Exceptions thrown, caught at boundaries, logged to console

**Patterns:**
- Python: try/catch in CLI commands, worker message handler
- TypeScript: Promise rejection handling, error state in React
- Cloud operations: Retry with exponential backoff (GCP SDK default)

## Cross-Cutting Concerns

**Logging:**
- Python: `logging` module with StreamHandler to stdout
- TypeScript: `console.log` for server logs
- Level: DEBUG in development

**Validation:**
- Input validation: CLI argument parsing, Express body parsing
- Data validation: DataFrame shape checks, column type inference

**Configuration:**
- YAML files for ML hyperparameters
- Environment variables for cloud credentials
- TypeScript path aliases for imports

---

*Architecture analysis: 2026-01-23*
*Update when major patterns change*
