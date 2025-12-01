
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import shap

model = joblib.load("gpa_predictor_model.pkl")
scaler = joblib.load("gpa_predictor_scaler.pkl")

app = FastAPI(title="SortED GPA Prediction API with SHAP")

class GPAInput(BaseModel):
    attendance_score: float
    assignment_score: float
    event_score: float
    academic_score: float
    spi: float

@app.post("/predict")
def predict_gpa(data: GPAInput):
    x = np.array([[
        data.attendance_score,
        data.assignment_score,
        data.event_score,
        data.academic_score,
        data.spi
    ]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return {"predicted_gpa": float(pred)}

@app.post("/explain")
def explain_gpa(data: GPAInput):
    x = np.array([[
        data.attendance_score,
        data.assignment_score,
        data.event_score,
        data.academic_score,
        data.spi
    ]])
    x_scaled = scaler.transform(x)

    explainer = shap.Explainer(model)
    shap_values = explainer(x_scaled)

    explanation = []
    feature_names = [
        "attendance_score",
        "assignment_score",
        "event_score",
        "academic_score",
        "spi"
    ]

    for i, name in enumerate(feature_names):
        explanation.append({
            "feature": name,
            "value": float(x[0][i]),
            "shap_value": float(shap_values.values[0][i])
        })

    return {
        "predicted_gpa": float(model.predict(x_scaled)[0]),
        "shap_explanation": explanation
    }
