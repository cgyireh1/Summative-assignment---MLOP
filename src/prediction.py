import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPrediction:
    def __init__(self, model_path, scaler_path, encoder_path=None):
        """
        Initializes the Prediction class with paths to the trained model, scaler, and encoder.

        Args:
            model_path (str): Path to the saved model file.
            scaler_path (str): Path to the saved scaler file.
            encoder_path (str, optional): Path to the saved label encoder file. Default is None.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path

        # Load the trained model and scaler
        self.model, self.scaler = self.load_model_and_scaler(model_path, scaler_path)

        # Load the encoder if provided
        if encoder_path:
            self.encoder = joblib.load(encoder_path)
        else:
            self.encoder = None

    def load_model_and_scaler(self, model_path, scaler_path):
        """
        Loads the trained model and scaler from disk.

        Args:
            model_path (str): Path to the saved model file.
            scaler_path (str): Path to the saved scaler file.

        Returns:
            tuple: A tuple containing:
                - model (sklearn.base.BaseEstimator): The trained model.
                - scaler (sklearn.preprocessing.StandardScaler): The scaler used to scale the features.
        """
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    def preprocess_new_data(self, new_data):
        """
        Preprocesses the new data to match the trained model's input format.

        Args:
            new_data (pandas.DataFrame): New data to be processed, must contain the same columns as the training data.

        Returns:
            numpy.ndarray: The processed feature matrix for the new data ready for prediction.
        """
        if self.encoder is not None:
            # Label encode the 'gender' column using the encoder
            new_data['gender'] = self.encoder.transform(new_data['gender'])

        # One-hot encode 'smoking_history' with all categories seen during training
        all_categories = [
            'smoking_history_current',
            'smoking_history_ever',
            'smoking_history_former',
            'smoking_history_never',
            'smoking_history_not current'
        ]
        
        # One-hot encode the 'smoking_history' column and align to training categories
        new_data_encoded = pd.get_dummies(new_data, columns=['smoking_history'])
        for category in all_categories:
            if category not in new_data_encoded:
                new_data_encoded[category] = 0  # Add missing category with 0

        # Ensure the column order matches the training data
        new_data_encoded = new_data_encoded[[
            'gender', 'age', 'hypertension', 'heart_disease', 'bmi',
            'HbA1c_level', 'blood_glucose_level'
        ] + all_categories]

        # Scale the features using the provided scaler
        X_new = self.scaler.transform(new_data_encoded)

        return X_new

    def predict_single(self, new_data):
        """
        Makes a prediction for a single data point.

        Args:
            new_data (pandas.DataFrame): The new data for which the prediction needs to be made.

        Returns:
            str: The prediction result, either "Diabetes" or "No Diabetes" based on the model's output.
        """
        # Preprocess the new data to match the model's expected input
        X_new = self.preprocess_new_data(new_data)

        # Make the prediction using the trained model
        prediction = self.model.predict(X_new)

        # Return "Diabetes" if the prediction is 1 (positive) else "No Diabetes"
        return "Diabetes" if prediction[0] == 1 else "No Diabetes"
