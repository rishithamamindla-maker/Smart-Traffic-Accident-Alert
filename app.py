from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/accident_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    speed = float(request.form["speed"])
    acceleration = float(request.form["acceleration"])
    impact = float(request.form["impact"])
    tilt = float(request.form["tilt"])

    # Use the exact feature names used during training
    input_data = pd.DataFrame([{
        "Speed": speed,
        "Acceleration": acceleration,
        "Impact": impact,
        "Tilt": tilt
    }])

    # Prediction
    prediction = model.predict(input_data)

    # Accident probability
    probability = model.predict_proba(input_data)[0][1]
    risk = probability * 100

    if prediction[0] == 1:
        result = "⚠️ Accident detected!"
        message = "🚨 Emergency alert should be triggered."
    else:
        result = "✅ No accident detected."
        message = "Vehicle appears to be safe."

    return render_template(
        "index.html",
        result=result,
        message=message,
        risk=f"{risk:.2f}",
        speed=speed,
        acceleration=acceleration,
        impact=impact,
        tilt=tilt
    )


if __name__ == "__main__":
    app.run(debug=True)