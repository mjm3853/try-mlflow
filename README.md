# try-mlflow

An MLflow experimentation repository for learning and testing MLflow's tracking, model registry, and artifact management capabilities. This project demonstrates MLflow workflows with practical examples using both classification and regression models.

## Overview

This repository contains example scripts showcasing different MLflow patterns:

- **Basic ML Tracking**: Iris classification with logistic regression
- **Experiment Management**: Creating experiments with rich metadata and tags
- **Complete ML Pipeline**: Apple sales forecasting with RandomForest regression
- **Data Generation**: Synthetic dataset creation with realistic features

## Features

- MLflow 3.1+ integration with modern best practices
- Local MLflow tracking server setup
- Model logging with comprehensive metrics
- Dataset lineage tracking
- Model registry integration
- Synthetic data generation with seasonality effects

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd try-mlflow
```

2. Install dependencies:
```bash
uv sync
```

## Usage

### Start MLflow UI

Start the MLflow tracking server (runs on http://127.0.0.1:5000 by default):

```bash
uv run mlflow ui
```

For custom port:
```bash
uv run mlflow ui --port 8080
```

### Generate Sample Data

Create synthetic apple sales dataset:

```bash
uv run python dataset.py
```

This generates `data/apple_sales.pkl` with features including seasonality, pricing effects, and promotional periods.

### Run Example Scripts

1. **Basic MLflow Tracking** (Iris Classification):
```bash
uv run python hello.py
```
Demonstrates basic MLflow tracking with sklearn LogisticRegression on the Iris dataset.

2. **Create Experiment with Metadata**:
```bash
uv run python create_experiment.py
```
Shows how to create experiments with comprehensive tags and business context.

3. **Complete ML Pipeline** (Apple Sales Prediction):
```bash
uv run python create_runs.py
```
Full ML pipeline with RandomForest regression, including data logging and model registration.

## Project Structure

```
try-mlflow/
├── README.md                 # This file
├── CLAUDE.md                # Claude Code guidance
├── pyproject.toml           # Python dependencies
├── uv.lock                  # Dependency lock file
├── data/                    # Generated datasets
│   └── apple_sales.pkl     # Synthetic apple sales data
├── mlruns/                  # MLflow tracking data
├── mlartifacts/            # MLflow model artifacts
├── hello.py                # Basic MLflow example
├── create_experiment.py    # Experiment creation example
├── create_runs.py          # Complete ML pipeline
└── dataset.py              # Synthetic data generation
```

## MLflow Integration Details

### Tracking Configuration
- **Tracking URI**: http://127.0.0.1:8080 (configurable)
- **Experiments**: Organized with descriptive names and comprehensive tagging
- **Runs**: Include parameters, metrics, artifacts, and model signatures

### Model Management
- **Model Registry**: Automatic model registration with versioning
- **Artifacts**: Models stored with input examples and signatures
- **Metadata**: Rich tagging for experiment organization

### Data Lineage
- **Dataset Logging**: Tracks data sources and versions
- **Input Tracking**: Links datasets to specific training runs
- **Reproducibility**: Complete parameter and environment tracking

## What You'll Learn

- Setting up MLflow tracking server
- Logging parameters, metrics, and artifacts
- Creating experiments with business context
- Model registration and versioning
- Dataset lineage tracking
- MLflow UI navigation and analysis
- Best practices for ML experiment organization

## Requirements

See `pyproject.toml` for complete dependency list. Main dependencies:
- mlflow>=3.1.0
- scikit-learn
- pandas
- numpy

## Contributing

This is a learning repository. Feel free to experiment with the code and add your own MLflow examples!
