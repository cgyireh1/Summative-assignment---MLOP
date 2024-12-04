from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from src.preprocessing import DataPreprocessing
from src.model import ModelPipeline
from src.prediction import DataPrediction

app = FastAPI()

# Paths
MODEL_PATH = "models/randomforest_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"
UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize prediction and retraining utilities
data_predictor = DataPrediction(MODEL_PATH, SCALER_PATH, ENCODER_PATH)
model_pipeline = ModelPipeline()


recent_file_path = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict")
async def predict(data: dict):
    """
    Predict diabetes for a single data point.
    """
    try:
        expected_features = ['gender', 'age', 'hypertension', 'heart_disease', 
                             'bmi', 'HbA1c_level', 'blood_glucose_level', 'smoking_history']

        # Check for missing features
        missing_features = [feature for feature in expected_features if feature not in data]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

        # Convert input data into a DataFrame with proper column names
        data_df = pd.DataFrame([data], columns=expected_features)

        print(f"Input data for prediction: \n{data_df}")

        # Make a prediction
        prediction_result = data_predictor.predict_single(data_df)

        if prediction_result == "Diabetes":
            prediction_message = "Diabetic. You should consult a doctor for a proper treatment plan 🫶."
        else:
            prediction_message = "Non-Diabetic. You do not have diabetes. Keep up the healthy lifestyle! 🎉"

        return {"prediction": prediction_message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload bulk data for retraining.
    """
    global recent_file_path
    try:
        # Add a timestamp to the uploaded file for easy tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, file_name)

        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        data_preprocessor = DataPreprocessing(file_path)
        if not data_preprocessor.validate_columns():
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="Uploaded file does not contain required columns."
            )

        # Update recent file path
        recent_file_path = file_path

        return {"message": "File uploaded successfully", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain-model")
async def retrain_model(file: UploadFile = File(...)):
    """
    Retrain the model using uploaded data.
    """
    global recent_file_path
    try:
        if not file:
            if not recent_file_path:
                raise HTTPException(status_code=400, detail="No file uploaded.")
            file_path = recent_file_path
        else:
            # Save the uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{timestamp}_{file.filename}"
            file_path = os.path.join(UPLOAD_DIR, file_name)

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Validate file
            data_preprocessor = DataPreprocessing(file_path)
            if not data_preprocessor.validate_columns():
                os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file does not contain required columns."
                )
        
        # Load and preprocess data
        data_preprocessor = DataPreprocessing(file_path)
        X, y = data_preprocessor.preprocess_data()

        class_distribution = y.value_counts().to_dict()

        # Handling class imbalance
        from sklearn.utils import resample
        X, y = resample(X, y, replace=True, n_samples=len(y), stratify=y, random_state=42)


        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        retrained_model, model_path = model_pipeline.retrain_model(X_train, y_train)

        acc, _, report = model_pipeline.evaluate_model(retrained_model, X_test, y_test)

        # Save updated model
        joblib.dump(retrained_model, MODEL_PATH)

        # Retraining results
        return JSONResponse(content={
            "accuracy": acc,
            "class_distribution": class_distribution,
            "classification_report": report
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/confusion-matrix")
async def get_confusion_matrix():
    """
    Retrieve the most recent confusion matrix.
    """
    global recent_cm_path
    if not recent_cm_path or not os.path.exists(recent_cm_path):
        raise HTTPException(status_code=404, detail="Recent confusion matrix not found.")
    return FileResponse(recent_cm_path)

# CORS Configuration
origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5501",
    "https://diabetes-prediction-web-app-l0ks.onrender.com"
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
