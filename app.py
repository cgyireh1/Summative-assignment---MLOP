import os
import joblib
import pandas as pd
import json
import shutil
from pydantic import BaseModel, Field
from fastapi import FastAPI, Form, UploadFile, Request, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from src.model import ModelPipeline
from src.prediction import DataPrediction
from src.preprocessing import DataPreprocessing
from datetime import datetime

app = FastAPI()


class FileUploadRequest(BaseModel):
    file: UploadFile = File(..., description="CSV file with data for retraining")
    retrain: bool = Form(False, description="Whether to retrain the model after file upload")

# Mount static files and pages
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/pages", StaticFiles(directory="pages"), name="pages")

templates = Jinja2Templates(directory="pages")

model_path = "models/randomforest_model.pkl"
scaler_path = "models/scaler.pkl"
encoder_path = "models/encoder.pkl"

# Initialize prediction object
predictor = DataPrediction(model_path=model_path, scaler_path=scaler_path, encoder_path=encoder_path)

# Path for the retraining logs
LOG_FILE_PATH = "logs/retraining_log.json"

# Create directories if they don't exist
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

#  the uploaded file
def save_uploaded_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

class PredictionRequest(BaseModel):
    gender: str = Field(..., example="e.g. Male/Female")
    age: int = Field(..., example="Enter age")
    hypertension: int = Field(..., example="e.g. 1 for Yes, 0 for No")
    heart_disease: int = Field(..., example="e.g. 1 for Yes, 0 for No")
    bmi: float = Field(..., example="e.g. 25")
    HbA1c_level: float = Field(..., example="e.g. 7.0")
    blood_glucose_level: float = Field(..., example="e.g. 100")
    smoking_history: str = Field(..., example="e.g. Never or Former or Current")

# Home Route (index.html)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})

# Model Prediction Endpoint
@app.post("/predict/") 
async def predict(request: PredictionRequest):
    new_data = pd.DataFrame([{
        "gender": request.gender,
        "age": request.age,
        "hypertension": request.hypertension,
        "heart_disease": request.heart_disease,
        "bmi": request.bmi,
        "HbA1c_level": request.HbA1c_level,
        "blood_glucose_level": request.blood_glucose_level,
        "smoking_history": request.smoking_history,
    }])

    try:
        print(f"Received data: {new_data}")  
        
        # Initialize predictor with correct paths
        predictor = DataPrediction(model_path=model_path, scaler_path=scaler_path, encoder_path=encoder_path)
        
        # Prediction
        prediction_result = predictor.predict_single(new_data)
        
        # Log the prediction result for debugging
        print(f"Prediction result: {prediction_result}")

        # Format the response based on prediction
        if prediction_result == 1:
            prediction_message = "Diabetic. You should consult a doctor for a proper treatment plan 🫶."
        else:
            prediction_message = "Non-Diabetes, You do not have diabetes. Keep up the healthy lifestyle! 🎉"
        
        return JSONResponse(content={"prediction": prediction_message})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"})

# Data Upload Page
@app.get("/upload_data/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "title": "Upload Data"})

# Data Upload Endpoint
@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...), retrain: str = Form("false")):
    retrain = retrain.lower() == "true"
    message = ""
    error = ""

    try:
        # Save the uploaded file with a timestamp to avoid overwriting
        file_location = f"static/uploads/{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        save_uploaded_file(file, file_location)

        # Validate file structure
        data_preprocessor = DataPreprocessing(file_location)
        if not data_preprocessor.validate_columns():
            error = "Invalid file structure. Please check the columns in the uploaded file."
            return JSONResponse(content={"error": error}, status_code=400)

        if retrain:
            # Log retraining information
            retraining_log = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset_used": file.filename,
                "model_path": model_path,
                "scaler_path": scaler_path,
            }

            # Append retraining logs
            if os.path.exists(LOG_FILE_PATH):
                with open(LOG_FILE_PATH, "r") as f:
                    retraining_logs = json.load(f)
            else:
                retraining_logs = []

            retraining_logs.append(retraining_log)
            with open(LOG_FILE_PATH, "w") as f:
                json.dump(retraining_logs, f, indent=4)

            # Retrain the model
            message = "File uploaded successfully, and retraining triggered."

            # Perform retraining and save model
            model_pipeline = ModelPipeline()
            X, y = data_preprocessor.preprocess_data()

            # Retrain the model
            trained_model = model_pipeline.retrain_model(X, y)

            # Save the retrained model
            model_pipeline.save_model(trained_model)
            model_pipeline.save_scaler()

        else:
            message = "File uploaded successfully"

        return JSONResponse(content={"message": message})

    except Exception as e:
        error = f"Error: {str(e)}"
        return JSONResponse(content={"error": error}, status_code=500)


# Retrain Model Page
@app.get("/retrain/", response_class=HTMLResponse)
async def retrain_page(request: Request):
    return templates.TemplateResponse("retrain.html", {"request": request, "title": "Retrain Model"})

@app.post("/retrain")
async def retrain_model(data_file: UploadFile = File(...), model_parameters: str = Form("default")):
    try:
        # Save the uploaded file
        file_location = f"temp_{data_file.filename}"
        with open(file_location, "wb") as file:
            file.write(data_file.file.read())

        # Load dataset
        data = pd.read_csv(file_location)
        
        X = data.drop("target", axis=1)
        y = data["target"]
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Initialize model with selected parameters
        if model_parameters == "tuned":
            model = RandomForestClassifier(n_estimators=200, max_depth=10)
        else:
            model = RandomForestClassifier()  # 
        # Retrain the model
        model.fit(X_train, y_train)

        # Save the retrained model
        model_filename = "retrained_model.pkl"
        joblib.dump(model, model_filename)
        
        os.remove(file_location)

        return JSONResponse(content={"message": "Model retrained successfully!"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})


# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:5000",
    "http://localhost:5501",
    "https://diabetes-prediction-web-app-l0ks.onrender.com" # for production
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
