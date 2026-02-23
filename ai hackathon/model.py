import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path
import joblib


MODEL_PATH = Path("trained_model.joblib")
DATA_PATH = Path("dataset.csv")


def load_data():
    """Load the synthetic dataset."""
    return pd.read_csv(DATA_PATH)


def train_model():
    """Train a simple RandomForest classifier on the dataset."""
    df = load_data()
    feature_cols = [
        "Age",
        "Gender",
        "SystolicBP",
        "DiastolicBP",
        "BloodSugar",
        "BMI",
        "Cholesterol",
        "Smoking",
        "ActivityLevel",
    ]
    target_col = "RiskLevel"

    X = df[feature_cols]
    y = df[target_col]

    # Categorical columns to encode
    categorical_cols = ["Gender", "Smoking", "ActivityLevel"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)

    # Display a quick report to console for transparency
    preds = clf.predict(X_test)
    print("Validation report:")
    print(classification_report(y_test, preds))

    joblib.dump(clf, MODEL_PATH)
    return clf


def load_model():
    """Load model from disk if available; otherwise train a new one."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


def predict_risk(
    model,
    age,
    gender,
    systolic_bp,
    diastolic_bp,
    blood_sugar,
    bmi,
    cholesterol,
    smoking,
    activity_level,
):
    """Run inference and return the predicted risk label."""
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
    return prediction


if __name__ == "__main__":
    # Allow quick retraining from CLI
    train_model()

