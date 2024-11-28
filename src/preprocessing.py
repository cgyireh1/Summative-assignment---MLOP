import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df, encoder=None):
    """Preprocesses the dataset: encoding and scaling."""
    if encoder is None:
        encoder = LabelEncoder()
        df['gender'] = encoder.fit_transform(df['gender'])
    else:
        df['gender'] = encoder.transform(df['gender'])

    # One-hot encode 'smoking_history'
    df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)

    # Define features (X) and target (y)
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, encoder


def split_data(X, y, test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
