import joblib

# Load trained model
model = joblib.load("model/accident_model.pkl")

print("🚗 Smart Traffic Accident Detection")
print("-----------------------------------")

# Get vehicle information
speed = float(input("Enter vehicle speed (km/h): "))
acceleration = float(input("Enter acceleration (m/s²): "))
impact = float(input("Enter impact level (0-10): "))
tilt = float(input("Enter vehicle tilt (degrees): "))

# Make prediction
prediction = model.predict([[speed, acceleration, impact, tilt]])

# Display result
if prediction[0] == 1:
    print("\n🚨 ACCIDENT DETECTED!")
    print("⚠️ Emergency alert should be generated.")
else:
    print("\n✅ No accident detected.")
    print("Vehicle appears to be safe.")