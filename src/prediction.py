import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model_and_scaler(model_path, scaler_path):
    """Load the trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_new_data(new_data, scaler, encoder):
    """Preprocess the new data to match the trained model's input."""
    # Label encode the 'gender' column
    new_data['gender'] = encoder.transform(new_data['gender'])

    # One-hot encode 'smoking_history' with all categories seen during training
    all_categories = [
        'smoking_history_current',
        'smoking_history_ever',
        'smoking_history_former',
        'smoking_history_never',
        'smoking_history_not current'
    ]
    
    # One-hot encode and align to training columns
    new_data_encoded = pd.get_dummies(new_data, columns=['smoking_history'])
    for category in all_categories:
        if category not in new_data_encoded:
            new_data_encoded[category] = 0  # Add missing category with 0

    # Ensure the column order matches the training data
    new_data_encoded = new_data_encoded[[
        'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level'
    ] + all_categories]

    # Scale the features
    X_new = scaler.transform(new_data_encoded)

    return X_new



def predict_single(new_data, model, scaler, encoder):
    """Make a prediction for a single data point."""
    # Preprocess the new data
    X_new = preprocess_new_data(new_data, scaler, encoder)

    # Make the prediction using the model
    prediction = model.predict(X_new)

    return "Diabetes" if prediction[0] == 1 else "No Diabetes"
