from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.schemas import ChurnPredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

DEFAULT_THRESHOLD = 0.24


class ModelArtifactsMissingError(FileNotFoundError):
    """Raised when model artifacts are missing."""


@lru_cache(maxsize=1)
def load_model_artifacts() -> tuple[Any, dict]:
    """
    Load model pipeline and metadata.

    Artifacts are cached after first load.
    """
    if not MODEL_PATH.exists():
        raise ModelArtifactsMissingError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python src/train_model.py` first."
        )

    if not METADATA_PATH.exists():
        raise ModelArtifactsMissingError(
            f"Model metadata not found at {METADATA_PATH}. "
            "Run `python src/train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


def get_model_metadata() -> dict:
    """
    Return model metadata.
    """
    _, metadata = load_model_artifacts()
    return metadata


def build_feature_frame(request: ChurnPredictionRequest) -> pd.DataFrame:
    """
    Convert API request into a single-row model feature frame.
    """
    payload = request.model_dump()

    feature_frame = pd.DataFrame([payload])

    return feature_frame


def classify_risk(probability: float, threshold: float) -> str:
    """
    Convert churn probability into a business risk level.
    """
    if probability >= 0.60:
        return "High"

    if probability >= threshold:
        return "Medium"

    return "Low"


def build_explanation(
    request: ChurnPredictionRequest,
    probability: float,
    threshold: float,
) -> list[str]:
    """
    Build simple business-readable explanation.

    This is not a SHAP or causal explanation. It summarizes known business
    risk indicators from the request.
    """
    explanations = []

    if request.Contract == "Month-to-month":
        explanations.append("Month-to-month contract increases churn risk.")

    if request.tenure <= 12:
        explanations.append("Low tenure indicates an early-life customer risk.")

    if request.InternetService == "Fiber optic":
        explanations.append("Fiber optic customers showed higher churn tendency in training data.")

    if request.OnlineSecurity == "No":
        explanations.append("No online security is associated with higher churn risk.")

    if request.TechSupport == "No":
        explanations.append("No tech support is associated with higher churn risk.")

    if request.PaymentMethod == "Electronic check":
        explanations.append("Electronic check payment method is associated with higher churn risk.")

    if request.PaperlessBilling == "Yes":
        explanations.append("Paperless billing is associated with higher churn tendency.")

    if request.Contract == "Two year":
        explanations.append("Two-year contract is a stabilizing factor.")

    if request.tenure >= 36:
        explanations.append("Longer tenure is a stabilizing factor.")

    if not explanations:
        explanations.append(
            "Prediction is based on the trained model probability and the configured decision threshold."
        )

    explanations.append(
        f"Predicted churn probability is {probability:.3f}; decision threshold is {threshold:.2f}."
    )

    return explanations


def predict_churn(request: ChurnPredictionRequest) -> dict:
    """
    Generate churn prediction response.
    """
    model, metadata = load_model_artifacts()

    threshold = float(metadata.get("decision_threshold", DEFAULT_THRESHOLD))

    feature_frame = build_feature_frame(request)

    probability = float(model.predict_proba(feature_frame)[:, 1][0])
    prediction = int(probability >= threshold)
    prediction_label = "Churn" if prediction == 1 else "No Churn"
    risk_level = classify_risk(probability, threshold)

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "prediction_label": prediction_label,
        "risk_level": risk_level,
        "decision_threshold": threshold,
        "explanation": build_explanation(
            request=request,
            probability=probability,
            threshold=threshold,
        ),
    }