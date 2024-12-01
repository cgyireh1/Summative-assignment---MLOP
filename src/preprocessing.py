import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.encoder = None
        self.scaler = None
        self.expected_columns = [
            "gender", "age", "hypertension", "heart_disease", "bmi",
            "HbA1c_level", "blood_glucose_level", "smoking_history", "diabetes"
        ]

    def validate_columns(self) -> bool:
        data = pd.read_csv(self.file_path)
        return all(col in data.columns for col in self.expected_columns)

    def data_info(self):
        return self.df.info()

    def describe_data(self):
        return self.df.describe()

    def check_missing_values(self):
        return self.df.isnull().sum()

    def preprocess_data(self):
        # Encode 'gender'
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.df['gender'] = self.encoder.fit_transform(self.df['gender'])
        else:
            self.df['gender'] = self.encoder.transform(self.df['gender'])

        self.df = pd.get_dummies(self.df, columns=['smoking_history'], drop_first=True)

        X = self.df.drop(columns=['diabetes']) 
        y = self.df['diabetes'] 

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled_df, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
