## Project Description

The **Diabetes Prediction Web Application** is a machine learning-based web app designed to predict diabetes risk using various health parameters such as age, BMI, blood glucose levels, and more. The app allows users to:
- Make predictions on diabetes risk based on individual health inputs.
- Upload datasets to retrain the machine learning model.

The web app is built using **FastAPI** for the backend, which serves the model predictions, and **HTML/CSS/JavaScript** for the frontend. Users can upload data in CSV format to retrain the model and improve its predictions.

---

## Features
- **Gender**: Male/Female
- **Age** 
- **hypertension** ()
- **heart_disease**()
- **bmi**
- **HbA1c_level**
- **blood_glucose_level**
- **smoking_history**
- **diabetes**
	
---

## Project Structure

```plaintext
diabetes_prediction-ML_Pipeline_Summative/
│
├── README.md 
|
├──requirements.txt
│
├── notebook/
│   └── diabetes_prediction.ipynb
│
├── src/ 
│   ├── preprocessing.py 
│   ├── model.py 
│   └── prediction.py 
│
├── data/ 
│   ├── train/  
│   └── test/
│
└── models/ 
    ├── randomforest_model.pkl 
    ├── encoder.pkl          
    └── scaler.pkl             
```
---

## URL

The web application is deployed and can be accessed at:

- **Live Web App URL**: [https://diabetes-prediction-web-app-l0ks.onrender.com](https://diabetes-prediction-web-app-l0ks.onrender.com)

---

## Requirements

- **Python 3.8+**
- **pip**
- **virtualenv**

---

## Setup Instructions

### Step 1: Clone the repository

Clone the project repository to your local machine using the following command:

```python
git clone https://github.com/cgyireh1/diabetes-prediction-web-app.git
```
cd diabetes-prediction-web-app

- Set up a virtual environment
Create and activate a virtual environment to manage dependencies:

```python
python3 -m venv venv
source venv/bin/activate
```

- Install the dependencies
Install all necessary Python packages using pip:
```python
- pip install -r requirements.txt
```

- Run the FastAPI server locally
To start the backend server, run the following command:
```python
uvicorn app:app --reload
```
This will start the server locally at http://localhost:8000.

## How to Use The app
- Prediction
1. Navigate to the Prediction Page.
2. Input the health data (e.g., age, BMI, blood glucose levels).
3. Click predict to get a prediction of diabetes risk.
- Upload Data for Retraining
1. Navigate to the Upload Data Page.
2. Upload a CSV file containing health data.
3. Optionally, check the box to trigger model retraining.
4. The system will automatically retrain the model and store the updated model file.

- Frontend Repo
https://github.com/cgyireh1/diabetes_prediction_web_app

