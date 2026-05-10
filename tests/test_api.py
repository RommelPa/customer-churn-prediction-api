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


def test_root_endpoint_returns_service_metadata():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Customer Churn Prediction API"
    assert data["docs"] == "/docs"
    assert data["predict"] == "/predict"


def test_health_endpoint_reports_model_available():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_available"] is True
    assert data["decision_threshold"] == 0.24


def test_predict_endpoint_returns_churn_prediction():
    response = client.post("/predict", json=SAMPLE_REQUEST)

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["churn_probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["No Churn", "Churn"]
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert data["decision_threshold"] == 0.24
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0


def test_predict_endpoint_rejects_invalid_contract():
    invalid_request = SAMPLE_REQUEST.copy()
    invalid_request["Contract"] = "Invalid contract"

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422