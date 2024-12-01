import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPrediction:
    def __init__(self, model_path, scaler_path, encoder_path=None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.model, self.scaler = self.load_model_and_scaler(model_path, scaler_path)

        # Load the encoder if provided
        if encoder_path:
            self.encoder = joblib.load(encoder_path)
        else:
            self.encoder = None

    def update_model_paths(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model, self.scaler = self.load_model_and_scaler(model_path, scaler_path)

    def load_model_and_scaler(self, model_path, scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    def preprocess_new_data(self, new_data):
        if self.encoder is not None:
            new_data['gender'] = self.encoder.transform(new_data['gender'])
        else:
            raise ValueError("Encoder for 'gender' is not provided.")
        
        all_categories = [
            'smoking_history_current',
            'smoking_history_ever',
            'smoking_history_former',
            'smoking_history_never',
            'smoking_history_not current'
        ]
        
        new_data_encoded = pd.get_dummies(new_data, columns=['smoking_history'], drop_first=False)
        for category in all_categories:
            if category not in new_data_encoded:
                new_data_encoded[category] = 0  
        feature_order = [
            'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
            'HbA1c_level', 'blood_glucose_level'
        ] + all_categories
        new_data_encoded = new_data_encoded[feature_order]

        X_new = self.scaler.transform(new_data_encoded)

        return X_new


    def predict_single(self, new_data):
        X_new = self.preprocess_new_data(new_data)
        prediction = self.model.predict(X_new)
        return "Diabetes" if prediction[0] == 1 else "No Diabetes"


# {
#   "gender": "Female",
#   "age": "25",
#   "hypertension": "1",
#   "heart_disease": "0",
#   "bmi": "25",
#   "HbA1c_level": "7.0",
#   "blood_glucose_level": "100",
#   "smoking_history": "Former"
# }