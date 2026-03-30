"""
Data Cleaning Module for Hospital Readmission Prediction
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class DataCleaner:
    """Handles data cleaning operations for the diabetic patient dataset."""
    
    def __init__(self, missing_threshold=0.5):
        """
        Initialize the DataCleaner.
        
        Args:
            missing_threshold (float): Threshold for dropping columns with missing values
        """
        self.missing_threshold = missing_threshold
        self.cols_to_drop = []
        self.categorical_imputer = None
        self.numerical_imputer = None
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df = df.copy()
        
        missing_placeholders = ['?', 'Unknown/Invalid', 'None', '']
        for col in df.columns:
            df[col] = df[col].replace(missing_placeholders, np.nan)
        
        missing_pct = df.isnull().sum() / len(df)
        self.cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        df = df.drop(columns=self.cols_to_drop)
        
        df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
        
        return df
    
    def impute_missing(self, df):
        """
        Impute remaining missing values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[list(categorical_cols)] = self.categorical_imputer.fit_transform(df[list(categorical_cols)])
        
        if len(numerical_cols) > 0:
            self.numerical_imputer = SimpleImputer(strategy='median')
            df[list(numerical_cols)] = self.numerical_imputer.fit_transform(df[list(numerical_cols)])
        
        return df
    
    def create_binary_target(self, df, target_col='readmitted'):
        """
        Create binary target variable.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: DataFrame with binary target
        """
        df = df.copy()
        df['readmitted_binary'] = (df[target_col] == '<30').astype(int)
        return df
    
    def encode_age(self, df, age_col='age'):
        """
        Encode age ranges as ordinal values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            age_col (str): Name of the age column
            
        Returns:
            pd.DataFrame: DataFrame with encoded age
        """
        age_mapping = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
            '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
            '[80-90)': 8, '[90-100)': 9
        }
        df['age_encoded'] = df[age_col].map(age_mapping)
        return df
    
    def select_features(self, df, numerical_features, categorical_features):
        """
        Select and prepare features for modeling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        
        exclude_cols = ['readmitted', 'readmitted_binary', 'age', 'diag_1', 'diag_2', 'diag_3']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df['readmitted_binary']
        
        return X, y, feature_cols
    
    def fit_transform(self, df):
        """
        Apply all cleaning steps.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (X, y, feature_cols) cleaned data
        """
        df = self.handle_missing_values(df)
        df = self.impute_missing(df)
        df = self.create_binary_target(df)
        df = self.encode_age(df)
        
        numerical_features = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications',
            'num_procedures', 'number_inpatient', 'number_outpatient',
            'number_emergency', 'num_diagnoses', 'precode'
        ]
        categorical_features = [
            'race', 'gender', 'discharge_disposition_id',
            'admission_source_id', 'admission_type_id'
        ]
        
        X, y, feature_cols = self.select_features(df, numerical_features, categorical_features)
        
        return X, y, feature_cols


def load_and_clean(filepath):
    """
    Convenience function to load and clean data.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, feature_cols) cleaned data
    """
    df = pd.read_csv(filepath)
    cleaner = DataCleaner()
    return cleaner.fit_transform(df)
