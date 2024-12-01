# Product Review Rating System

A machine learning pipeline for automatically rating product reviews using RoBERTa, built with Kedro and MLflow.

## Overview

This system processes product reviews and predicts rating scores (1-5) using a fine-tuned RoBERTa model. It includes data validation, model training, monitoring, and deployment capabilities.

## Comprehensive Documentation

Quarto docs available from `docs_quarto/_site/index.html`

## Features

- **Data Processing**: Handles CSV review data with text cleaning and preprocessing
- **Model**: Fine-tuned RoBERTa model for 5-class classification
- **MLflow Integration**: Experiment tracking and model versioning
- **Data Monitoring**: Drift detection and data quality validation using Evidently
- **API Server**: FastAPI-based prediction endpoint with logging
- **Testing**: Nodes tested using pytest which run with every build
- **Docker Support**: Containerized model serving

## Main Project Structure

```
src/review_rating/pipelines
                    ├── data_processing/
                    │   ├── dataset.py       # PyTorch dataset implementation
                    │   └── nodes.py         # Data processing pipeline nodes
                    │   └── pipeline.py
                    ├── data_science/
                    │   └── nodes.py         # Model training and evaluation
                    │   └── pipeline.py
                    ├── data_monitoring
                    │   └── nodes.py
                    │   └── pipeline.py
```

## Requirements

- Python 3.12
- PyTorch
- Transformers
- Kedro
- MLflow
- FastAPI
- Docker
- Quarto
- Pytest

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the prediction server:
```bash
docker build -t review-rating-server .
docker run -d -p 5002:5002 --name review-rating-server review-rating-server
```

3. Run the pipeline:
```bash
kedro run
```

## API Endpoints

- `POST /predict`: Get rating prediction for a review
- `GET /batch-predict`: Get rating predictions for a list of reviews
- `GET /health`: Chech if the server is ready
- `GET /logs`: Query model prediction logs

## Model Details

- Architecture: RoBERTa-base
- Task: Multi-class classification (5 classes)
- Input: Review text (max length: 512 tokens)
- Output: Rating prediction (1-5)

## Monitoring

The system includes data monitoring capabilities:
- Data drift detection
- Missing value analysis
- Duplicate detection
- Data quality validation