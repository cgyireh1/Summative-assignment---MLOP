import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_random_forest(X_train, y_train):
    """Trains a Random Forest model."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model with test data."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report

def save_model(model, file_path):
    """Saves the trained model to a file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Loads a trained model from a file."""
    return joblib.load(file_path)


def retrain_model(X_train, y_train, model_path='models/random_forest_model.pkl'):
    """Retrains the model with new data."""
    # Train a new model
    model = train_random_forest(X_train, y_train)
    
    # Save the retrained model
    save_model(model, model_path)
    
    return model

