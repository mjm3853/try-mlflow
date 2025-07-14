# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an MLflow experimentation repository for learning and testing MLflow's tracking, model registry, and artifact management capabilities. The project uses `uv` for Python dependency management and contains example scripts demonstrating different MLflow workflows.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Start MLflow UI (default port 5000)
uv run mlflow ui

# Start MLflow UI on custom port
uv run mlflow ui --port 8080

# Generate synthetic dataset
uv run python dataset.py
```

### Running Examples
```bash
# Basic MLflow tracking example (Iris classification)
uv run python hello.py

# Create experiments with metadata
uv run python create_experiment.py

# Run training with full MLflow integration (Apple sales prediction)
uv run python create_runs.py
```

## Project Architecture

### Core Components

**MLflow Configuration**: The project uses a local MLflow tracking server typically running on `http://127.0.0.1:8080`. All scripts are configured to use this tracking URI.

**Experiment Structure**: 
- `hello.py` demonstrates basic MLflow tracking with the Iris dataset using LogisticRegression
- `create_experiment.py` shows how to create experiments with rich metadata and tags
- `create_runs.py` implements a complete ML pipeline for apple sales forecasting using RandomForestRegressor

**Data Management**: 
- `dataset.py` contains synthetic data generation for apple sales prediction with seasonality, promotions, and inflation effects
- Data is stored as pickle files in the `data/` directory
- MLflow's dataset logging is used to track data lineage

### MLflow Integration Patterns

**Experiment Organization**: Uses descriptive experiment names with comprehensive tagging including project metadata, team information, and business context.

**Model Logging**: All models are logged with:
- Complete parameter tracking
- Multiple evaluation metrics
- Model signatures for input/output validation
- Input examples for model serving
- Automatic model registration

**Artifact Management**: Models and datasets are stored in the local `mlruns/` and `mlartifacts/` directories with organized experiment hierarchy.

## Data Flow

1. **Data Generation**: `dataset.py` creates synthetic apple sales data with realistic features including seasonality, pricing effects, and promotional periods
2. **Experiment Setup**: `create_experiment.py` establishes experiments with business-relevant metadata
3. **Model Training**: `create_runs.py` demonstrates the complete ML lifecycle from data loading to model registration
4. **Model Serving**: Trained models can be loaded as PyFunc models for inference

The codebase emphasizes MLflow best practices including proper experiment organization, comprehensive logging, and model lifecycle management suitable for production environments.