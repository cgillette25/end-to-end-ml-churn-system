---
title: End-to-End ML Churn System
emoji: 📉
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# End-to-End ML Churn System

**Live demo:**  
https://huggingface.co/spaces/cgillette25/end-to-end-ml-churn-system

**Source code:**  
https://github.com/cgillette25/end-to-end-ml-churn-system

---

## Overview

This project demonstrates a **production-style, end-to-end machine learning system** built to reflect the responsibilities of a Machine Learning Engineer.

The system covers the full ML lifecycle:
- data ingestion and preprocessing
- model training and evaluation
- API-based inference
- user-facing UI
- containerized cloud deployment

The goal is not model novelty, but **system correctness, structure, and deployability**.

---

## Architecture
Training Pipeline
↓
Trained Model Artifact
↓
FastAPI Inference Service
↓
Streamlit UI
↓
Docker Container
↓
Hugging Face Spaces


Key design principles:
- strict feature validation at inference time
- clear separation between training and serving
- deterministic, reproducible deployment

---

## Model

- **Dataset:** IBM Telco Customer Churn
- **Task:** Binary classification (churn vs. no churn)
- **Best model:** Histogram Gradient Boosting
- **Evaluation metrics:**
  - ROC AUC ≈ 0.85
  - F1, Precision, Recall
  - Confusion Matrix

Model selection and tuning were performed using cross-validation, with the final model persisted as a reusable inference artifact.

---

## API

The inference service is exposed via FastAPI.

### Endpoints
- `GET /health`  
  Health check and model version reporting

- `POST /predict`  
  Returns churn probability and binary prediction

Strict request validation is enforced using Pydantic schemas to prevent schema drift between training and inference.

---

## User Interface

The Streamlit UI provides:
- an interactive form matching the trained feature schema
- probabilistic churn predictions
- lightweight interpretation messaging
- live API health status

The UI communicates exclusively with the FastAPI service, mirroring a real client–server setup.

---

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- FastAPI
- Pydantic
- Streamlit
- Docker
- Hugging Face Spaces

---

## Run Locally

```bash
docker build -t churn-app .
docker run -p 7860:7860 churn-app
```

The application will be available at:
http://localhost:7860