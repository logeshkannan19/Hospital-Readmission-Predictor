"""
Modeling Module for Hospital Readmission Prediction.

This module provides utilities for training and evaluating machine learning models.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib


class ReadmissionModel:
    """
    Handles model training and prediction for hospital readmission.
    
    Attributes:
        model_type: Type of model to train.
        test_size: Proportion of test set.
        random_state: Random seed for reproducibility.
    """
    
    def __init__(
        self, 
        model_type: str = 'gradient_boosting', 
        test_size: float = 0.3, 
        random_state: int = 42
    ) -> None:
        """
        Initialize the ReadmissionModel.
        
        Args:
            model_type: Type of model ('gradient_boosting', 'xgboost', 'rf', 'lr').
            test_size: Test set proportion.
            random_state: Random seed.
        """
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.smote: Optional[SMOTE] = None
        self.feature_cols: Optional[list] = None
        self.metrics: Dict[str, float] = {}
        
    def prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for modeling.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        val_size = 0.15 / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=self.random_state, 
            stratify=y_train
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_scaled, y_train)
        
        return (
            X_train_resampled, 
            X_val_scaled, 
            X_test_scaled,
            y_train_resampled, 
            y_val, 
            y_test
        )
    
    def create_model(self) -> None:
        """Create the model based on model_type."""
        if self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100, 
                    max_depth=6, 
                    learning_rate=0.1,
                    random_state=self.random_state, 
                    eval_metric='logloss'
                )
            except ImportError:
                self.model = GradientBoostingClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=self.random_state
                )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state, 
                n_jobs=-1
            )
        elif self.model_type == 'lr':
            self.model = LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000, 
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            Training metrics dictionary.
        """
        self.create_model()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        return self.metrics
    
    def tune_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        param_grid: Optional[Dict[str, list]] = None
    ) -> None:
        """
        Tune model hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            param_grid: Parameter grid for tuning.
        """
        if param_grid is None:
            if self.model_type in ['gradient_boosting', 'xgboost']:
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [4, 5, 6],
                    'learning_rate': [0.05, 0.1]
                }
            elif self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [100, 150],
                    'max_depth': [8, 10],
                    'min_samples_split': [2, 5]
                }
            else:
                param_grid = {'C': [0.1, 1, 10]}
        
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5, 
            scoring='roc_auc', 
            n_jobs=-1, 
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.metrics['best_params'] = grid_search.best_params_
        self.metrics['best_cv_score'] = grid_search.best_score_
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Tuple of (test_metrics, predictions, probabilities).
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return test_metrics, y_pred, y_pred_proba
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            feature_names: List of feature names.
            
        Returns:
            DataFrame with feature importances.
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return pd.DataFrame()
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Tuple of (predictions, probabilities).
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model and related objects.
        
        Args:
            filepath: Base filepath for saving.
        """
        joblib.dump(self.model, f"{filepath}_model.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.metrics, f"{filepath}_metrics.pkl")
        
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Base filepath for loading.
        """
        self.model = joblib.load(f"{filepath}_model.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.metrics = joblib.load(f"{filepath}_metrics.pkl")


def train_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    model_type: str = 'gradient_boosting'
) -> Tuple[ReadmissionModel, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to train a readmission model.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        model_type: Type of model to train.
        
    Returns:
        Tuple of (trained model, test data).
    """
    model = ReadmissionModel(model_type=model_type)
    
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
    
    model.train(X_train, y_train, X_val, y_val)
    
    return model, (X_test, y_test)
