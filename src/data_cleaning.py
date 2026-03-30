"""
Data Cleaning Module for Hospital Readmission Prediction.

This module provides utilities for cleaning and preprocessing the diabetic patient dataset.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class DataCleaner:
    """
    Handles data cleaning operations for the diabetic patient dataset.
    
    Attributes:
        missing_threshold: Threshold for dropping columns with missing values.
        cols_to_drop: List of columns dropped due to high missing percentage.
    """
    
    def __init__(self, missing_threshold: float = 0.5) -> None:
        """
        Initialize the DataCleaner.
        
        Args:
            missing_threshold: Threshold for dropping columns (default: 0.5).
        """
        self.missing_threshold = missing_threshold
        self.cols_to_drop: List[str] = []
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.numerical_imputer: Optional[SimpleImputer] = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        
        missing_placeholders = ['?', 'Unknown/Invalid', 'None', '']
        for col in df.columns:
            df[col] = df[col].replace(missing_placeholders, np.nan)
        
        missing_pct = df.isnull().sum() / len(df)
        self.cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        if self.cols_to_drop:
            df = df.drop(columns=self.cols_to_drop)
        
        id_columns = ['encounter_id', 'patient_nbr']
        df = df.drop(columns=[c for c in id_columns if c in df.columns], errors='ignore')
        
        return df
    
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute remaining missing values.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        if numerical_cols:
            self.numerical_imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
        
        return df
    
    def create_binary_target(self, df: pd.DataFrame, target_col: str = 'readmitted') -> pd.DataFrame:
        """
        Create binary target variable.
        
        Args:
            df: Input DataFrame.
            target_col: Name of the target column.
            
        Returns:
            DataFrame with binary target.
        """
        df = df.copy()
        df['readmitted_binary'] = (df[target_col] == '<30').astype(int)
        return df
    
    def encode_age(self, df: pd.DataFrame, age_col: str = 'age') -> pd.DataFrame:
        """
        Encode age ranges as ordinal values.
        
        Args:
            df: Input DataFrame.
            age_col: Name of the age column.
            
        Returns:
            DataFrame with encoded age.
        """
        age_mapping: Dict[str, int] = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
            '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
            '[80-90)': 8, '[90-100)': 9
        }
        df['age_encoded'] = df[age_col].map(age_mapping)
        return df
    
    def select_features(
        self, 
        df: pd.DataFrame, 
        numerical_features: List[str], 
        categorical_features: List[str]
    ) -> tuple:
        """
        Select and prepare features for modeling.
        
        Args:
            df: Input DataFrame.
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
            
        Returns:
            Tuple of (X, y, feature_cols).
        """
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        
        exclude_cols = ['readmitted', 'readmitted_binary', 'age', 'diag_1', 'diag_2', 'diag_3']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df['readmitted_binary']
        
        return X, y, feature_cols
    
    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Apply all cleaning steps.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Tuple of (X, y, feature_cols).
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


def load_and_clean(filepath: str) -> tuple:
    """
    Convenience function to load and clean data.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Tuple of (X, y, feature_cols).
    """
    df = pd.read_csv(filepath)
    cleaner = DataCleaner()
    return cleaner.fit_transform(df)
