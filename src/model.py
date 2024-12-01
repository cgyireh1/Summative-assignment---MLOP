import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ModelPipeline:
    def __init__(self, model_dir='models/models', scaler_path='models/scaler.pkl'):
        """
        Initializes the ModelPipeline class with a directory to save models and scaler.
        """
        self.model_dir = model_dir
        self.scaler_path = scaler_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def preprocess_data(self, X, y, scaler=None):
        """
        Preprocess the input data: handle missing values, scale features, etc.
        """
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X) 
        
        return X_scaled, y, scaler

    def train_random_forest(self, X_train, y_train):
        """
        Trains a Random Forest model.
        """
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        return rf_model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the trained model and generates metrics.
        """
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, cm, report

    def plot_confusion_matrix(self, cm, labels=['No Diabetes', 'Diabetes']):
        """
        Plots the confusion matrix.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def save_model(self, model):
        """
        Saves the trained model with a unique name.
        """
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('retrained_model_')]
        model_numbers = [int(f.split('_')[2].split('.')[0]) for f in model_files if f.split('_')[2].split('.')[0].isdigit()]
        next_model_number = max(model_numbers, default=0) + 1

        model_filename = os.path.join(self.model_dir, f'retrained_model_{next_model_number}.pkl')

        # Save model using joblib
        joblib.dump(model, model_filename)

        print(f"Model successfully saved as {model_filename}")
        return model_filename

    def load_model(self, file_path):
        """
        Loads a trained model from a specified file.
        """
        return joblib.load(file_path)

    def load_scaler(self):
        """
        Loads the scaler from the saved scaler file.
        """
        if os.path.exists(self.scaler_path):
            return joblib.load(self.scaler_path)
        else:
            return None

    def retrain_model(self, X_train, y_train, model_path=None):
        """
        Retrains the Random Forest model with new data and saves the retrained model.
        """
        # Load the existing scaler
        scaler = self.load_scaler()

        # Preprocess the data
        X_train, y_train, scaler = self.preprocess_data(X_train, y_train, scaler)

        if model_path:
            model = self.load_model(model_path)
            print(f"Loaded existing model from {model_path}")
        else:
            model = self.train_random_forest(X_train, y_train)

        # Retrain the model with new data
        model.fit(X_train, y_train)

        model_filename = self.save_model(model)

        return model, model_filename
