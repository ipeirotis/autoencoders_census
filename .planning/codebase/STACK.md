# Technology Stack

**Analysis Date:** 2026-01-23

## Languages

**Primary:**
- Python 3.9+ - ML pipeline, data processing, Vertex AI training (`main.py`, `worker.py`, `train/task.py`)
- TypeScript 5.9 - Frontend application and Express server (`frontend/`)

**Secondary:**
- JavaScript - Build scripts, config files (`postcss.config.js`)

## Runtime

**Environment:**
- Python 3.10-slim (Docker container for Vertex AI) - `Dockerfile`
- Node.js (Vite targets Node 22) - `frontend/vite.config.server.ts`

**Package Manager:**
- npm with `package-lock.json` - `frontend/package-lock.json`
- pip with `requirements.txt` - `requirements.txt`

## Frameworks

**Core:**
- React 18.2.0 - UI framework - `frontend/package.json`
- Express 5.2.1 - Backend API server - `frontend/server/index.ts`
- TensorFlow 2.15.1 - Deep learning framework - `requirements.txt`
- Keras 2.15.0 - Neural network API - `requirements.txt`

**Testing:**
- Vitest - Frontend unit tests - `frontend/client/lib/utils.spec.ts`
- unittest - Python standard library tests - `tests/`

**Build/Dev:**
- Vite 5.0.0 - Frontend bundling and dev server - `frontend/vite.config.ts`
- TypeScript 5.9.3 - Type safety - `frontend/tsconfig.json`
- Tailwind CSS 3.4.19 - Styling - `frontend/tailwind.config.ts`

## Key Dependencies

**Critical (Python ML Stack):**
- tensorflow==2.15.1 - Deep learning - `requirements.txt`
- keras==2.15.0 - Neural networks - `requirements.txt`
- keras-tuner==1.4.8 - Hyperparameter optimization - `requirements.txt`
- pandas==2.2.3 - Data manipulation - `requirements.txt`
- scikit-learn==1.5.2 - ML utilities - `requirements.txt`
- numpy==1.26.4 - Numerical computing - `requirements.txt`

**Critical (Google Cloud):**
- google-cloud-storage==3.7.0 - File storage - `requirements.txt`
- google-cloud-firestore==2.22.0 - Document database - `requirements.txt`
- google-cloud-pubsub==2.34.0 - Message queue - `requirements.txt`
- google-cloud-aiplatform==1.132.0 - Vertex AI training - `requirements.txt`

**Infrastructure (Frontend):**
- @google-cloud/storage ^7.18.0 - GCS client - `frontend/package.json`
- @google-cloud/firestore ^8.0.0 - Firestore client - `frontend/package.json`
- @google-cloud/pubsub ^5.2.0 - Pub/Sub client - `frontend/package.json`
- @tanstack/react-query ^5.90.12 - Data fetching - `frontend/package.json`

**UI Components:**
- @radix-ui/react-* - Accessible UI primitives - `frontend/package.json`
- lucide-react ^0.562.0 - Icons - `frontend/package.json`
- sonner ^2.0.7 - Toast notifications - `frontend/package.json`

## Configuration

**Environment:**
- `.env` files with environment variables
- Required: `GOOGLE_CLOUD_PROJECT`, `GCS_BUCKET_NAME`, `PUBSUB_SUBSCRIPTION_ID`
- Optional: `API_KEY`, `DATABASE_URL`, `PORT`, `HOST`

**Build:**
- `frontend/tsconfig.json` - TypeScript compiler with path aliases (@/*, @shared/*)
- `frontend/vite.config.ts` - Vite bundler configuration
- `frontend/tailwind.config.ts` - Tailwind with dark mode support

**ML Pipeline:**
- `config/simple_autoencoder.yaml` - Standard AE hyperparameters
- `config/simple_variational_autoencoder.yaml` - VAE hyperparameters
- `config/hp_autoencoder.yaml` - Tuning search space

## Platform Requirements

**Development:**
- macOS/Linux/Windows with Python 3.9+ and Node.js
- Docker for containerized ML training
- Google Cloud SDK for GCP access

**Production:**
- Vertex AI for ML training (Docker container)
- Google Cloud Storage for file storage
- Firestore for metadata and results
- Pub/Sub for job queuing

---

*Stack analysis: 2026-01-23*
*Update after major dependency changes*
