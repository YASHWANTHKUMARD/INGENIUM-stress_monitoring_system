from flask import Flask, render_template, request
from model import load_model, predict_risk

app = Flask(__name__)

# Load model and encoders once at startup
model_bundle = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    explanation = None
    recommendations = None

    if request.method == "POST":
        try:
            # Grab and sanitize form inputs
            age = float(request.form.get("age", 0))
            gender = request.form.get("gender", "Male")
            systolic = float(request.form.get("systolic_bp", 0))
            diastolic = float(request.form.get("diastolic_bp", 0))
            blood_sugar = float(request.form.get("blood_sugar", 0))
            bmi = float(request.form.get("bmi", 0))
            cholesterol = float(request.form.get("cholesterol", 0))
            smoking = request.form.get("smoking", "No")
            activity = request.form.get("activity_level", "Moderate")

            prediction = predict_risk(
                model_bundle,
                age=age,
                gender=gender,
                systolic_bp=systolic,
                diastolic_bp=diastolic,
                blood_sugar=blood_sugar,
                bmi=bmi,
                cholesterol=cholesterol,
                smoking=smoking,
                activity_level=activity,
            )

            explanation = {
                "Low": "Your metrics are mostly within healthy ranges.",
                "Medium": "Some metrics are elevated. Monitoring and lifestyle tweaks are advised.",
                "High": "Multiple risk factors detected. Consult a healthcare professional.",
            }.get(prediction, "Risk level unavailable.")

            recommendations = {
                "Low": [
                    "Maintain balanced diet and regular activity.",
                    "Keep routine health check-ups.",
                    "Stay hydrated and sleep 7-8 hours.",
                ],
                "Medium": [
                    "Increase physical activity to at least 150 minutes/week.",
                    "Reduce sugar and saturated fats; prioritize vegetables.",
                    "Monitor blood pressure and glucose regularly.",
                ],
                "High": [
                    "Consult a doctor for personalized guidance.",
                    "Adopt a heart-healthy diet (DASH/Mediterranean).",
                    "Avoid smoking and limit alcohol.",
                ],
            }.get(prediction, [])

        except ValueError:
            prediction = "Invalid input"
            explanation = "Please enter valid numeric values."
            recommendations = []

    return render_template(
        "index.html",
        prediction=prediction,
        explanation=explanation,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    # Debug mode for local development; remove debug=True for production
    app.run(host="0.0.0.0", port=5000, debug=True)

