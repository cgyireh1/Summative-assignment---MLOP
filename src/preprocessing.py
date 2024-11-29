import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self, file_path):
        """
        Initialize the DataPreprocessing class with a dataset file path.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.encoder = None
        self.scaler = None

    def data_info(self):
        """
        Returns a concise summary of the DataFrame structure and data types.

        :return: Summary of the DataFrame information.
        :rtype: None
        """
        return self.df.info()

    def describe_data(self):
        """
        Returns a summary of the statistical measures of the DataFrame.

        :return: A pandas DataFrame containing summary statistics of the DataFrame.
        :rtype: pandas.DataFrame
        """
        return self.df.describe()

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame.

        :return: A pandas Series containing the count of missing values for each column.
        :rtype: pandas.Series
        """
        return self.df.isnull().sum()

    def preprocess_data(self):
        """
        Preprocess the dataset:
        - Encodes categorical variables.
        - Scales numerical features.

        Returns:
            tuple:
                - X (numpy.ndarray): Scaled feature matrix.
                - y (pandas.Series): Target variable ('diabetes').
                - scaler (StandardScaler): The scaler used for feature scaling.
                - encoder (LabelEncoder): The encoder used for categorical encoding.
        """
        # Encode 'gender'
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.df['gender'] = self.encoder.fit_transform(self.df['gender'])
        else:
            self.df['gender'] = self.encoder.transform(self.df['gender'])

        # One-hot encode 'smoking_history'
        self.df = pd.get_dummies(self.df, columns=['smoking_history'], drop_first=True)

        # Define features (X) and target (y)
        X = self.df.drop(columns=['diabetes'])  # Features excluding the target
        y = self.df['diabetes']  # Target variable

        # Scale features using StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled_df, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y (numpy.ndarray or pandas.Series): Target variable.
            test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
            random_state (int): Seed for reproducibility of the split (default is 42).

        Returns:
            tuple:
                - X_train (numpy.ndarray): Training feature matrix.
                - X_test (numpy.ndarray): Testing feature matrix.
                - y_train (numpy.ndarray): Training target variable.
                - y_test (numpy.ndarray): Testing target variable.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
