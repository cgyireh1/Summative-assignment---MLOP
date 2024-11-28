from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocessing import preprocess_data
from prediction import load_model_and_scaler, predict_single
import os

app = Flask(__name__)

# Load model and scaler
model, scaler = load_model_and_scaler('models/random_forest_model.pkl', 'models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles predictions for a single data point."""
    data = request.json
    new_data = pd.DataFrame([data])
    prediction = predict_single(new_data, model, scaler)
    return jsonify({'prediction': prediction})

@app.route('/upload', methods=['POST'])
def upload_data():
    """Uploads new data to retrain the model."""
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        X, y, scaler, encoder = preprocess_data(df)
        retrained_model = retrain_model(X, y)
        return jsonify({'message': 'Model retrained successfully!'})
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
