from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
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

    prediction = model.predict(
        [[speed, acceleration, impact, tilt]]
    )[0]

    if prediction == 1:
        result = "🚨 ACCIDENT DETECTED!"
        status = "danger"
    else:
        result = "✅ NO ACCIDENT DETECTED"
        status = "safe"

    return render_template(
        "index.html",
        result=result,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)