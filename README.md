# 🚗 Smart Traffic Accident Alert System

A machine learning-based system that predicts the possibility of a traffic accident using vehicle sensor parameters.

## Features

- Vehicle speed analysis
- Acceleration analysis
- Impact level detection
- Vehicle tilt analysis
- Machine learning prediction
- Flask web application

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML/CSS
- Git & GitHub

## Project Structure

Smart-Traffic-Accident-Alert/
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── traffic_data.csv
│
├── model/
│   ├── accident_model.pkl
│   └── train_model.py
│
└── templates/
    └── index.html

## Installation

Clone the repository:

git clone https://github.com/rishithamamindla-maker/Smart-Traffic-Accident-Alert.git

Go to the project folder:

cd Smart-Traffic-Accident-Alert

Create virtual environment:

python -m venv venv

Activate virtual environment:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Train the Model

python model/train_model.py

## Run the Application

python app.py

Open:

http://127.0.0.1:5000

## Machine Learning Model

The system uses a Random Forest Classifier to predict accident possibility based on vehicle parameters.

## Future Improvements

- Real-time GPS integration
- IoT sensor integration
- Emergency SMS alerts
- Google Maps integration
- Real-time accident detection
- Cloud deployment

## Author

Rishitha Mamindla