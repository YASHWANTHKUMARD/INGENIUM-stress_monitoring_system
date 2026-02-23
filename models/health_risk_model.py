"""
Health risk prediction model integration.

This module reuses the AI Hackathon project model located under
`ai hackathon/` and exposes simple helper functions that can be
used from the FastAPI backend.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
AI_HACKATHON_DIR = BASE_DIR / "ai hackathon"
MODEL_PATH = AI_HACKATHON_DIR / "trained_model.joblib"
DATA_PATH = AI_HACKATHON_DIR / "dataset.csv"

_health_model: Any | None = None


def _load_model_from_disk():
    """Load model from disk; train is handled in original project."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Health model file not found at {MODEL_PATH}. "
            "Run the training script in the 'ai hackathon' folder first."
        )
    return joblib.load(MODEL_PATH)


def get_health_model():
    """Lazy singleton accessor for the health risk model."""
    global _health_model
    if _health_model is None:
        _health_model = _load_model_from_disk()
    return _health_model


def predict_health_risk(
    *,
    age: float,
    gender: str,
    systolic_bp: float,
    diastolic_bp: float,
    blood_sugar: float,
    bmi: float,
    cholesterol: float,
    smoking: str,
    activity_level: str,
) -> str:
    """Run inference and return the predicted risk label."""
    model = get_health_model()
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                "BloodSugar": blood_sugar,
                "BMI": bmi,
                "Cholesterol": cholesterol,
                "Smoking": smoking,
                "ActivityLevel": activity_level,
            }
        ]
    )
    prediction = model.predict(input_df)[0]
    return str(prediction)


