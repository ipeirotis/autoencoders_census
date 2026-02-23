# CLAUDE.md

## Project Overview

AutoEncoder Outlier Detection Platform -- a full-stack ML system that detects problematic entries (outliers) in survey and tabular data using autoencoders. Users upload CSV files via a web UI and receive outlier analysis powered by TensorFlow and optionally Google Cloud Vertex AI.

The core idea: train an autoencoder to reconstruct survey responses; rows with high reconstruction error are likely from inattentive or mischievous respondents.

## Repository Structure

```
autoencoders_census/
├── main.py                  # CLI entry point (click commands: train, evaluate, find_outliers, etc.)
├── worker.py                # Pub/Sub worker that dispatches Vertex AI training jobs
├── utils.py                 # Seed setting, model save/load, plotting, evaluation helpers
├── dataset/
│   └── loader.py            # DataLoader: loads CSVs, bins numeric vars, applies "Rule of 9" filter
├── features/
│   └── transform.py         # Table2Vector: one-hot encoding for categorical, MinMax for numeric
├── model/
│   ├── factory.py           # get_model() factory -- returns AE, VAE, or PCA model
│   ├── autoencoder.py       # AutoencoderModel (standard AE with per-attribute softmax decoder)
│   ├── variational_autoencoder.py  # VariationalAutoencoderModel (VAE with Gaussian/Gumbel prior)
│   ├── pcamodel.py          # LinearAutoencoder (PCA baseline)
│   ├── base.py              # VAE base class with reconstruction_loss, kl_loss helpers
│   ├── loss.py              # Custom loss functions (percentile-adjusted categorical crossentropy)
│   └── layers.py            # Custom Keras layers (sampling layer for VAE)
├── train/
│   ├── trainer.py           # Trainer: model.fit with EarlyStopping, hyperparameter search via Keras Tuner
│   └── task.py              # Vertex AI task entry point
├── evaluate/
│   ├── evaluator.py         # Evaluator: reconstruction accuracy, confusion matrices, ROC AUC
│   ├── outliers.py          # get_outliers_list(): computes per-row reconstruction error
│   └── generator.py         # Generator: synthetic sample generation from trained VAE decoder
├── config/                  # YAML configs for model hyperparameters
│   ├── simple_autoencoder.yaml
│   ├── simple_variational_autoencoder.yaml
│   ├── hp_autoencoder.yaml              # Hyperparameter search space for AE
│   └── hp_variational_autoencoder.yaml  # Hyperparameter search space for VAE
├── data/                    # CSV datasets (gitignored except checked-in SADC files)
├── tests/                   # Unit tests (pytest/unittest)
│   ├── model/test_autoencoder.py
│   ├── model/test_loss.py
│   ├── features/test_transform.py
│   └── dataset/test_loader.py
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── frontend/                # Full web UI (React + Express + TypeScript)
│   ├── client/              # React 18 frontend (Vite, Tailwind, shadcn/ui)
│   └── server/              # Express.js API server (file upload, GCS signed URLs, Firestore)
└── requirements.txt         # Python dependencies
```

## Key Concepts

### Rule of 9
Columns with more than 9 unique values or only 1 unique value are dropped before training. This keeps only low-cardinality categorical variables suitable for the autoencoder.

### Data Pipeline
1. **Load** -- `DataLoader` reads CSV, drops/renames/selects columns per dataset config
2. **Clean** -- Fill NaN with "missing", convert to string, apply Rule of 9 filter
3. **Vectorize** -- `Table2Vector` one-hot encodes categorical columns
4. **Train** -- Autoencoder learns to reconstruct the one-hot vectors
5. **Score** -- Reconstruction error per row = outlier score (higher = more anomalous)

### Models
- **AE** (`AutoencoderModel`): Standard autoencoder with per-attribute softmax outputs and `CustomCategoricalCrossentropyAE` loss (with percentile-based loss trimming)
- **VAE** (`VariationalAutoencoderModel`): Variational autoencoder supporting Gaussian and Gumbel-Softmax priors
- **PCA** (`LinearAutoencoder`): PCA baseline for comparison

### Supported Datasets
Built-in dataset configs in `main.py` and `utils.py`: `sadc_2017`, `sadc_2015`, `pennycook_1`, `pennycook_2`, `pennycook`, `bot_bot_mturk`, `inattentive`, `attention_check`, `moral_data`, `mturk_ethics`, `public_opinion`, `racial_data`. Custom CSVs can be uploaded via the web frontend.

## Build & Run

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- TensorFlow 2.15.1 (not compatible with Apple Silicon -- use `tensorflow-macos` locally)

### Python Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CLI Commands
```bash
# Train a model
python main.py train --model_name AE --data sadc_2017 --config config/simple_autoencoder.yaml

# Hyperparameter search
python main.py search_hyperparameters --model_name AE --config config/hp_autoencoder.yaml

# Evaluate reconstruction accuracy
python main.py evaluate --model_path cache/simple_model/autoencoder --data sadc_2017

# Find outliers
python main.py find_outliers --model_path cache/simple_model/autoencoder --data sadc_2017
```

### Web UI (3 terminals)
```bash
# Terminal 1: Python worker (needs GCP credentials)
python worker.py

# Terminal 2: Express API server
cd frontend && npm install && npm run dev:server   # http://localhost:5001

# Terminal 3: React dev server
cd frontend && npm run dev                          # http://localhost:5173
```

### Running Tests
```bash
python -m pytest tests/ -v
```

## Development Guidelines

### Code Style
- Python code uses standard library logging, click for CLI, and type hints sparingly
- Model configs are YAML files in `config/`
- Frontend uses TypeScript with Tailwind CSS and shadcn/ui components

### Architecture Patterns
- `DataLoader.prepare_original_dataset()` returns `(clean_df, metadata_dict)` where metadata contains `variable_types` and `ignored_columns`
- The `train` and `find_outliers` CLI commands duplicate data-cleaning logic inline (the `run_training_pipeline()` function in `main.py` provides a reusable version)
- Custom Keras losses are registered via `@keras.utils.register_keras_serializable()`
- Models are saved in TensorFlow SavedModel format (`save_format="tf"`)

### Common Gotchas
- `DataLoader.load_data()` return format varies: some loaders return `(df, variable_types_dict)`, others return `(df, metadata_dict)` with nested structure. The `train` and `find_outliers` commands handle both via isinstance checks.
- The `test_loader.py` tests reference `DataLoader.DATASET_URL_2015` class attributes that no longer exist -- these tests will fail.
- `*.csv` files are gitignored. The `data/` directory ships with `sadc_2015only_national.csv` and `sadc_2017only_national_full.csv` which are tracked.
- The frontend requires GCP credentials (`.env` files) to function. Without them, only the CLI pipeline works.
- TensorFlow 2.15.1 requires AVX instructions; Apple Silicon Macs need `tensorflow-macos` instead.

### Environment Variables (for web UI / cloud features)
```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GCS_BUCKET_NAME=your-bucket-name
PUBSUB_SUBSCRIPTION_ID=job-upload-topic-sub
```
