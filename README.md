# Customer Churn Prediction API: Serving Machine Learning Models with FastAPI

## Overview

This project builds a FastAPI service for customer churn prediction.

The goal is to convert a trained machine learning model into a usable prediction API with input validation, clear response structure, reproducible model training, automated API tests, and Docker-based execution.

This project demonstrates the full path from raw customer data to cleaned training data, trained model artifact, API endpoint, validated prediction response, and containerized service execution.

---

## Business Context

Customer churn prediction helps companies identify customers who are likely to leave.

A churn prediction API can support retention workflows, CRM systems, customer success dashboards, proactive outreach campaigns, customer risk scoring, and operational decision systems.

The API returns not only a churn prediction, but also a churn probability, risk level, decision threshold, and simple business-readable explanation.

---

## Main Objectives

- Train a reproducible churn prediction model.
- Save a machine learning pipeline artifact.
- Build a FastAPI inference service.
- Validate request inputs with Pydantic.
- Return churn probability, prediction, risk level, and explanation.
- Add health and metadata endpoints.
- Add automated API tests.
- Add Docker support for local containerized execution.
- Document local usage, Docker usage, and reproducibility.

---

## Dataset

This project uses the Telco Customer Churn dataset.

Expected raw file:

```text
data/raw/telco_customer_churn.csv
```

Expected columns:

| Column | Description |
|---|---|
| `customerID` | Customer identifier |
| `gender` | Customer gender |
| `SeniorCitizen` | Senior citizen flag |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months with the company |
| `PhoneService` | Whether the customer has phone service |
| `MultipleLines` | Multiple phone line status |
| `InternetService` | Internet service type |
| `OnlineSecurity` | Online security service status |
| `OnlineBackup` | Online backup service status |
| `DeviceProtection` | Device protection service status |
| `TechSupport` | Tech support service status |
| `StreamingTV` | Streaming TV service status |
| `StreamingMovies` | Streaming movies service status |
| `Contract` | Contract type |
| `PaperlessBilling` | Paperless billing status |
| `PaymentMethod` | Payment method |
| `MonthlyCharges` | Monthly customer charges |
| `TotalCharges` | Total customer charges |
| `Churn` | Target variable |

Raw data is not included in this repository. Place `telco_customer_churn.csv` inside `data/raw/` before training the model.

---

## Project Structure

```text
customer-churn-prediction-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── data/
│   ├── raw/
│   └── processed/
├── examples/
│   ├── sample_request.json
│   └── sample_response.json
├── models/
├── src/
│   └── train_model.py
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── .dockerignore
├── Dockerfile
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Model Training

The model is trained using `GradientBoostingClassifier`.

The training script builds a full scikit-learn pipeline with numeric imputation, numeric scaling, categorical imputation, one-hot encoding, and gradient boosting classification.

Train the model with:

```bash
python src/train_model.py
```

The script generates local artifacts:

```text
models/churn_model_pipeline.joblib
models/model_metadata.json
data/processed/telco_customer_churn_clean.csv
```

These files are intentionally ignored by Git.

---

## Target Distribution

| Churn | Count | Percent |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

The positive class is customer churn.

---

## Feature Groups

### Numeric features

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

### Categorical features

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

---

## Model Performance

Validation metrics using threshold `0.24`:

| Metric | Value |
|---|---:|
| Accuracy | 0.7395 |
| Precision | 0.5058 |
| Recall | 0.8128 |
| F1-score | 0.6236 |
| ROC-AUC | 0.8433 |
| Average Precision | 0.6600 |
| Validation rows | 1,409 |
| Validation churn rate | 0.2654 |
| Customers flagged | 601 |
| Churners captured | 304 |
| Churners missed | 70 |

---

## Decision Threshold

The API uses a decision threshold of `0.24`.

This threshold prioritizes recall. The model is designed to capture more potential churners, even if that creates more false positives.

This is appropriate for a retention workflow where missing a high-risk customer can be more costly than contacting some customers who may not churn.

---

## API Endpoints

### Root endpoint

```http
GET /
```

Returns service metadata.

Example response:

```json
{
  "service": "Customer Churn Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

### Health endpoint

```http
GET /health
```

Returns model availability status.

Example response:

```json
{
  "status": "ok",
  "model_available": true,
  "model_name": "gradient_boosting_churn_pipeline",
  "decision_threshold": 0.24
}
```

### Metadata endpoint

```http
GET /metadata
```

Returns model metadata, features, threshold, and validation metrics.

### Prediction endpoint

```http
POST /predict
```

Returns churn prediction for one customer.

---

## Example Request

File:

```text
examples/sample_request.json
```

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

---

## Example Response

File:

```text
examples/sample_response.json
```

```json
{
  "churn_probability": 0.405,
  "prediction": 1,
  "prediction_label": "Churn",
  "risk_level": "Medium",
  "decision_threshold": 0.24,
  "explanation": [
    "Month-to-month contract increases churn risk.",
    "Low tenure indicates an early-life customer risk.",
    "No online security is associated with higher churn risk.",
    "No tech support is associated with higher churn risk.",
    "Electronic check payment method is associated with higher churn risk.",
    "Paperless billing is associated with higher churn tendency.",
    "Predicted churn probability is 0.405; decision threshold is 0.24."
  ]
}
```

---

## Risk Levels

| Risk Level | Rule |
|---|---|
| Low | Probability below threshold |
| Medium | Probability greater than or equal to threshold and below 0.60 |
| High | Probability greater than or equal to 0.60 |

This makes API responses easier to consume in CRM or dashboard systems.

---

## Explanation Logic

The API returns simple business-readable explanations based on known churn risk indicators, such as month-to-month contract, low tenure, fiber optic service, no online security, no tech support, electronic check payment, and paperless billing.

These explanations are not SHAP values and are not causal explanations. They are simple business rules that help make the model output more understandable.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/RommelPa/customer-churn-prediction-api.git
cd customer-churn-prediction-api
```

### 2. Create a virtual environment

```bash
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add raw data

Place the dataset here:

```text
data/raw/telco_customer_churn.csv
```

### 5. Train model

```bash
python src/train_model.py
```

### 6. Run API

```bash
uvicorn app.main:app --reload
```

### 7. Open Swagger docs

```text
http://127.0.0.1:8000/docs
```

---

## Docker Usage

This project can also run inside a Docker container.

Before building the image, train the model locally so the required model artifacts exist:

```bash
python src/train_model.py
```

The Docker image copies the local `models/` directory into the container.

### Build Docker image

Recommended command for Docker Desktop environments:

```bash
docker buildx build --load -t customer-churn-prediction-api:local .
```

Alternative command:

```bash
docker build -t customer-churn-prediction-api:local .
```

If `docker run` cannot find the image after using `docker build`, use the `buildx --load` command.

### Verify local image

```bash
docker image ls | findstr customer
```

Expected image name:

```text
customer-churn-prediction-api:local
```

### Run Docker container

```bash
docker run --rm -p 8000:8000 customer-churn-prediction-api:local
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

### Test health endpoint

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/health" `
  -Method Get
```

Expected response:

```text
status: ok
model_available: True
model_name: gradient_boosting_churn_pipeline
decision_threshold: 0.24
```

### Test prediction endpoint

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -InFile "examples/sample_request.json"
```

Expected response fields:

```text
churn_probability
prediction
prediction_label
risk_level
decision_threshold
explanation
```

---

## PowerShell Prediction Example

With the API running locally or inside Docker:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -InFile "examples/sample_request.json"
```

Expected response fields:

```text
churn_probability
prediction
prediction_label
risk_level
decision_threshold
explanation
```

---

## Automated Tests

Run tests with:

```bash
pytest
```

Current test coverage includes:

- root endpoint,
- health endpoint,
- prediction endpoint,
- invalid request validation.

Expected result:

```text
4 passed
```

---

## Reproducibility Check

Before considering the project valid, run:

```bash
python -m py_compile app/main.py app/predictor.py app/schemas.py src/train_model.py tests/test_api.py
python src/train_model.py
pytest
uvicorn app.main:app --reload
```

Then test the API using Swagger or PowerShell.

---

## Docker Reproducibility Check

After the local model is trained, run:

```bash
docker buildx build --load -t customer-churn-prediction-api:local .
docker image ls | findstr customer
docker run --rm -p 8000:8000 customer-churn-prediction-api:local
```

In another terminal:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/health" `
  -Method Get
```

Then:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -InFile "examples/sample_request.json"
```

Expected result:

- `/health` returns `status: ok`;
- `/predict` returns churn probability, prediction label, risk level, threshold, and explanation.

---

## Tools Used

- Python
- FastAPI
- Pydantic
- scikit-learn
- pandas
- numpy
- joblib
- Uvicorn
- pytest
- httpx
- Docker
- Git
- GitHub

---

## Design Decisions

### Why FastAPI?

FastAPI provides fast local API development, automatic OpenAPI documentation, request validation, and strong compatibility with Python ML workflows.

### Why Pydantic?

Pydantic validates request fields before they reach the model. This prevents invalid categories, invalid numeric ranges, and malformed payloads.

### Why Gradient Boosting?

Gradient Boosting provides strong tabular classification performance without requiring heavy infrastructure.

### Why threshold 0.24?

The threshold prioritizes recall for retention use cases. The goal is to flag more potential churners, accepting more false positives.

### Why not commit model artifacts?

Model artifacts and processed data are generated locally and ignored by Git. This keeps the repository lightweight and forces reproducibility from source data.

### Why Docker?

Docker packages the API runtime, Python dependencies, application code, examples, and locally trained model artifacts into a container image.

This makes the service easier to run outside the original development environment.

---

## Limitations

- The model is trained on a static public Telco churn dataset.
- The API is designed for local inference, not production deployment.
- The explanation field is rule-based and not a formal model interpretability method.
- The current API predicts one customer per request.
- The model artifact must be generated locally before serving predictions.
- The Docker image copies local model artifacts; it does not train the model during image build.
- There is no authentication, logging, monitoring, or database integration in this version.
- The model should be retrained and validated before use in a real business environment.

---

## Next Steps

Possible extensions:

- add batch prediction endpoint,
- add Docker Compose support,
- deploy to a cloud service,
- add model versioning,
- add SHAP-based explanations,
- add request logging,
- add monitoring for data drift,
- add CI workflow for tests,
- add authentication for private deployment,
- optimize Docker image size.

---

## Spanish Summary

Este proyecto construye una API de predicción de churn con FastAPI.

La API convierte un modelo de machine learning entrenado en un servicio local usable. Recibe datos de un cliente y devuelve probabilidad de churn, predicción binaria, nivel de riesgo, umbral de decisión y una explicación simple.

El modelo usa Gradient Boosting con un pipeline reproducible de scikit-learn. El umbral de decisión es 0.24, priorizando recall para capturar más clientes en riesgo.

El proyecto incluye entrenamiento reproducible, validación de inputs con Pydantic, endpoints `/health`, `/metadata` y `/predict`, pruebas automatizadas con pytest, ejemplos de request/response y ejecución local con Docker.