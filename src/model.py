import os
import pickle as pk
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ModelPipeline:
    def __init__(self, model_dir='models/models'):
        """
        Initializes the ModelPipeline class with a directory to save models.

        Args:
            model_dir (str, optional): Directory where models will be saved. Default is 'models/models'.
        """
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def preprocess_data(self, X, y):
        """
        Preprocess the input data: handle missing values, scale features, etc.
        
        Args:
            X (DataFrame): Features data.
            y (Series): Target data.
        
        Returns:
            tuple: Processed feature data and target data
        """
        
        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y

    def train_random_forest(self, X_train, y_train):
        """
        Trains a Random Forest model.

        Args:
            X_train (array-like): Feature matrix for training data.
            y_train (array-like): Target vector for training data.

        Returns:
            RandomForestClassifier: Trained RandomForestClassifier model.
        """
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        return rf_model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the trained model and generates metrics.

        Args:
            model (RandomForestClassifier): The trained model to evaluate.
            X_test (array-like): Feature matrix for test data.
            y_test (array-like): True labels for test data.

        Returns:
            tuple: Contains accuracy score, confusion matrix, and classification report.
        """
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, cm, report

    def plot_confusion_matrix(self, cm, labels=['No Diabetes', 'Diabetes']):
        """
        Plots the confusion matrix.

        Args:
            cm (ndarray): Confusion matrix.
            labels (list, optional): Labels for the confusion matrix. Default is ['No Diabetes', 'Diabetes'].
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


    def save_model(self, model):
        """
        Saves the trained model with a unique name, following the pattern 'retrained_model_{number}.pkl'.

        Args:
            model (RandomForestClassifier): The trained model to be saved.

        Returns:
            str: The path to the saved model file.
        """
        # List existing model files with the naming pattern
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('retrained_model_')]

        # Determine the next model number to maintain unique naming
        model_numbers = [
            int(f.split('_')[2].split('.')[0]) for f in model_files if f.split('_')[2].split('.')[0].isdigit()
        ]
        next_model_number = max(model_numbers, default=0) + 1

        # Construct the filename with a unique model number
        model_filename = os.path.join(self.model_dir, f'retrained_model_{next_model_number}.pkl')

        # Save the model using pickle
        with open(model_filename, 'wb') as file:
            pk.dump(model, file)

        print(f"Model successfully saved as {model_filename}")
        return model_filename

    def load_model(self, file_path):
        """
        Loads a trained model from a specified file.

        Args:
            file_path (str): The path to the saved model file.

        Returns:
            RandomForestClassifier: The loaded model.
        """
        return joblib.load(file_path)

    def retrain_model(self, X_train, y_train, model_path=None):
        """
        Retrains the Random Forest model with new data and saves the retrained model.

        Args:
            X_train (array-like): Feature matrix for training data.
            y_train (array-like): Target vector for training data.
            model_path (str, optional): Path to an existing model to retrain. If None, a new model will be created.

        Returns:
            RandomForestClassifier: The retrained model.
        """
        # Preprocess the data
        X_train, y_train = self.preprocess_data(X_train, y_train)

        # Check if an existing model needs to be loaded
        if model_path:
            # Load the existing model
            model = self.load_model(model_path)
            print(f"Loaded existing model from {model_path}")
        else:
            # Train a new model from scratch
            model = self.train_random_forest(X_train, y_train)

        # retrain the model with the new data
        model.fit(X_train, y_train)

        # Save the retrained model
        self.save_model(model)

        return model
