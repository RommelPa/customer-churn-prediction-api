from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "telco_customer_churn_clean.csv"
MODEL_PATH = MODELS_DIR / "churn_model_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

TARGET_COLUMN = "Churn"
TARGET_LABEL_COLUMN = "churn_label"

DECISION_THRESHOLD = 0.24
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}.\n\n"
            "Place telco_customer_churn.csv inside data/raw/."
        )

    data = pd.read_csv(path, dtype={"customerID": "str", "TotalCharges": "str"})

    return data


def validate_raw_columns(data: pd.DataFrame) -> None:
    """
    Validate expected raw dataset columns.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Telco churn data for model training.

    TotalCharges contains blank strings for customers with zero tenure.
    Those values are converted to 0.0.
    """
    data = raw_data.copy()

    validate_raw_columns(data)

    duplicate_customer_ids = data["customerID"].duplicated().sum()

    if duplicate_customer_ids > 0:
        raise ValueError(
            f"Duplicate customerID values found: {duplicate_customer_ids}"
        )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].replace(" ", np.nan),
        errors="coerce",
    )

    zero_tenure_missing_total = (
        data["TotalCharges"].isna() & (data["tenure"] == 0)
    )

    data.loc[zero_tenure_missing_total, "TotalCharges"] = 0.0

    remaining_missing_total_charges = data["TotalCharges"].isna().sum()

    if remaining_missing_total_charges > 0:
        raise ValueError(
            "TotalCharges still contains missing values after cleaning: "
            f"{remaining_missing_total_charges}"
        )

    data[TARGET_LABEL_COLUMN] = data[TARGET_COLUMN].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    if data[TARGET_LABEL_COLUMN].isna().any():
        raise ValueError("Target column contains unexpected Churn labels.")

    return data


def build_model_pipeline() -> Pipeline:
    """
    Build the churn prediction pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float,
) -> dict:
    """
    Evaluate model using probability threshold.
    """
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(y_validation, predictions, zero_division=0),
        "recall": recall_score(y_validation, predictions, zero_division=0),
        "f1": f1_score(y_validation, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_validation, probabilities),
        "average_precision": average_precision_score(y_validation, probabilities),
        "validation_rows": int(len(y_validation)),
        "validation_churn_rate": float(y_validation.mean()),
        "customers_flagged": int(predictions.sum()),
        "churners_captured": int(((predictions == 1) & (y_validation == 1)).sum()),
        "churners_missed": int(((predictions == 0) & (y_validation == 1)).sum()),
    }

    return metrics


def save_artifacts(
    model: Pipeline,
    clean_dataset: pd.DataFrame,
    metrics: dict,
) -> None:
    """
    Save model pipeline and metadata.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    clean_dataset.to_csv(CLEAN_DATA_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "gradient_boosting_churn_pipeline",
        "model_type": "GradientBoostingClassifier",
        "decision_threshold": DECISION_THRESHOLD,
        "target_column": TARGET_COLUMN,
        "target_label_column": TARGET_LABEL_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_rows": int(len(clean_dataset)),
        "churn_rate": float(clean_dataset[TARGET_LABEL_COLUMN].mean()),
        "metrics": metrics,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def print_training_summary(clean_dataset: pd.DataFrame, metrics: dict) -> None:
    """
    Print model training summary.
    """
    print("=" * 80)
    print("CUSTOMER CHURN MODEL TRAINING SUMMARY")
    print("=" * 80)

    print("\n1. Dataset")
    print(f"Rows: {len(clean_dataset):,}")
    print(f"Columns: {clean_dataset.shape[1]:,}")

    print("\n2. Target distribution")
    target_distribution = (
        clean_dataset[TARGET_COLUMN]
        .value_counts(normalize=False)
        .rename_axis("Churn")
        .reset_index(name="count")
    )
    target_distribution["percent"] = (
        target_distribution["count"] / len(clean_dataset) * 100
    ).round(2)
    print(target_distribution.to_string(index=False))

    print("\n3. Feature groups")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(NUMERIC_FEATURES)
    print(f"\nCategorical features: {len(CATEGORICAL_FEATURES)}")
    print(CATEGORICAL_FEATURES)

    print("\n4. Validation metrics")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    print("\n5. Saved artifacts")
    print(f"Clean data: {CLEAN_DATA_PATH}")
    print(f"Model pipeline: {MODEL_PATH}")
    print(f"Model metadata: {METADATA_PATH}")


if __name__ == "__main__":
    raw_churn_data = load_raw_data()
    clean_churn_data = clean_data(raw_churn_data)

    X = clean_churn_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = clean_churn_data[TARGET_LABEL_COLUMN]

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    churn_model = build_model_pipeline()
    churn_model.fit(X_train, y_train)

    validation_metrics = evaluate_model(
        model=churn_model,
        X_validation=X_validation,
        y_validation=y_validation,
        threshold=DECISION_THRESHOLD,
    )

    final_model = build_model_pipeline()
    final_model.fit(X, y)

    save_artifacts(
        model=final_model,
        clean_dataset=clean_churn_data,
        metrics=validation_metrics,
    )

    print_training_summary(
        clean_dataset=clean_churn_data,
        metrics=validation_metrics,
    )