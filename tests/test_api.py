from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


SAMPLE_REQUEST = {
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
    "TotalCharges": 29.85,
}


MOCK_PREDICTION = {
    "churn_probability": 0.405,
    "prediction": 1,
    "prediction_label": "Churn",
    "risk_level": "Medium",
    "decision_threshold": 0.24,
    "explanation": [
        "Month-to-month contract increases churn risk.",
        "Predicted churn probability is 0.405; decision threshold is 0.24.",
    ],
}


MOCK_METADATA = {
    "model_name": "gradient_boosting_churn_pipeline",
    "decision_threshold": 0.24,
}


def test_root_endpoint_returns_service_metadata():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Customer Churn Prediction API"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"
    assert data["predict"] == "/predict"


def test_health_endpoint_reports_model_available(monkeypatch):
    monkeypatch.setattr(
        "app.main.get_model_metadata",
        lambda: MOCK_METADATA,
    )

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_available"] is True
    assert data["model_name"] == "gradient_boosting_churn_pipeline"
    assert data["decision_threshold"] == 0.24


def test_metadata_endpoint_returns_model_metadata(monkeypatch):
    monkeypatch.setattr(
        "app.main.get_model_metadata",
        lambda: MOCK_METADATA,
    )

    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "gradient_boosting_churn_pipeline"
    assert data["decision_threshold"] == 0.24


def test_predict_endpoint_returns_churn_prediction(monkeypatch):
    monkeypatch.setattr(
        "app.main.predict_churn",
        lambda request: MOCK_PREDICTION,
    )

    response = client.post("/predict", json=SAMPLE_REQUEST)

    assert response.status_code == 200

    data = response.json()

    assert data["churn_probability"] == 0.405
    assert data["prediction"] == 1
    assert data["prediction_label"] == "Churn"
    assert data["risk_level"] == "Medium"
    assert data["decision_threshold"] == 0.24
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0


def test_predict_endpoint_rejects_invalid_contract():
    invalid_request = SAMPLE_REQUEST.copy()
    invalid_request["Contract"] = "Invalid contract"

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422