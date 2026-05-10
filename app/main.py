from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from app.predictor import (
    ModelArtifactsMissingError,
    get_model_metadata,
    predict_churn,
)
from app.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
)


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "FastAPI service for customer churn prediction using a trained "
        "machine learning pipeline."
    ),
    version="1.0.0",
)


@app.get("/")
def root() -> dict:
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        metadata = get_model_metadata()

        return HealthResponse(
            status="ok",
            model_available=True,
            model_name=metadata.get("model_name"),
            decision_threshold=metadata.get("decision_threshold"),
        )

    except ModelArtifactsMissingError:
        return HealthResponse(
            status="model_missing",
            model_available=False,
            model_name=None,
            decision_threshold=None,
        )


@app.get("/metadata")
def metadata() -> dict:
    try:
        return get_model_metadata()

    except ModelArtifactsMissingError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest) -> ChurnPredictionResponse:
    try:
        prediction = predict_churn(request)

        return ChurnPredictionResponse(**prediction)

    except ModelArtifactsMissingError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error