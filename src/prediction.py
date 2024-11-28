import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_model_and_scaler(model_path, scaler_path):
    """Load the trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_new_data(new_data, scaler, encoder):
    """Preprocess the new data to match the trained model's input."""
    # Ensure the 'gender' column is encoded using the same encoder
    new_data['gender'] = encoder.transform(new_data['gender'])

    # Ensure consistent one-hot encoding
    new_data = pd.get_dummies(new_data, columns=['smoking_history'], drop_first=True)
    
    # Ensure the number of columns matches the training data
    new_data = new_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale the features using the loaded scaler
    X_new = scaler.transform(new_data)

    return X_new


def predict_single(new_data, model, scaler, encoder):
    """Make a prediction for a single data point."""
    # Preprocess the new data
    X_new = preprocess_new_data(new_data, scaler, encoder)

    # Make the prediction using the model
    prediction = model.predict(X_new)

    return "Diabetes" if prediction[0] == 1 else "No Diabetes"
