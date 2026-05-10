from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
PhoneService = Literal["Yes", "No"]
MultipleLines = Literal["Yes", "No", "No phone service"]
InternetService = Literal["DSL", "Fiber optic", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
Contract = Literal["Month-to-month", "One year", "Two year"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class ChurnPredictionRequest(BaseModel):
    gender: Gender
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: PhoneService
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    prediction_label: Literal["No Churn", "Churn"]
    risk_level: Literal["Low", "Medium", "High"]
    decision_threshold: float
    explanation: list[str]


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_name: str | None = None
    decision_threshold: float | None = None