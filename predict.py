import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/accident_model.pkl")

print("🚗 Smart Traffic Accident Detection")
print("-----------------------------------")

speed = float(input("Enter vehicle speed (km/h): "))
acceleration = float(input("Enter acceleration (m/s²): "))
impact = float(input("Enter impact level (0-10): "))
tilt = float(input("Enter vehicle tilt (degrees): "))

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

print(f"\n📊 Accident Risk: {risk:.2f}%")

if prediction[0] == 1:
    print("\n⚠️ Accident detected!")
    print("🚨 Emergency alert should be triggered.")
else:
    print("\n✅ No accident detected.")
    print("Vehicle appears to be safe.")