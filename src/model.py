import os
import pickle as pk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

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

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return acc, cm, report

def save_model(model, model_dir='models/models', verbose=True):
    """
    Save the trained model with a unique name following the pattern 'retrained_model_{number}.pkl'.

    Args:
        model: The trained model to be saved.
        model_dir (str): Directory where the model will be saved.
        verbose (bool): Whether to print a confirmation message. Defaults to True.

    Returns:
        str: The filename of the saved model.
    """
    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # List existing model files with the naming pattern
    model_files = [f for f in os.listdir(model_dir) if f.startswith('retrained_model_')]

    # Determine the next model number
    model_numbers = [
        int(f.split('_')[2].split('.')[0]) for f in model_files if f.split('_')[2].split('.')[0].isdigit()
    ]
    next_model_number = max(model_numbers, default=0) + 1

    # Construct the filename
    model_filename = os.path.join(model_dir, f'retrained_model_{next_model_number}.pkl')

    # Save the model to the file
    with open(model_filename, 'wb') as file:
        pk.dump(model, file)

    # Print confirmation if verbose is True
    if verbose:
        print(f"Model successfully saved as {model_filename}")

    return model_filename



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