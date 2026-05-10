# Customer Churn Prediction API

## Overview

This project builds a FastAPI service for customer churn prediction.

The goal is to convert a trained machine learning model into a usable prediction API with input validation, clear outputs, reproducible training, and API-level tests.

## Business Context

Customer churn prediction helps companies identify customers at risk of leaving.

A prediction API can support retention workflows, CRM systems, dashboards, and customer success operations.

## Objectives

- Train a reproducible churn prediction model.
- Save a machine learning pipeline artifact.
- Build a FastAPI inference service.
- Validate request inputs with Pydantic.
- Return churn probability, prediction, risk level, and business explanation.
- Add API tests.
- Document local usage.

## Project Structure

```text
customer-churn-prediction-api/
├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── data/
│   ├── raw/
│   └── processed/
├── examples/
│   └── sample_request.json
├── models/
├── src/
│   └── train_model.py
├── tests/
│   └── test_api.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Status

Project in progress.