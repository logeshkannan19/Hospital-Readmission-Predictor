"""
Modeling Module for Hospital Readmission Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE
import joblib


class ReadmissionModel:
    """Handles model training and prediction for hospital readmission."""
    
    def __init__(self, model_type='xgboost', test_size=0.15, random_state=42):
        """
        Initialize the ReadmissionModel.
        
        Args:
            model_type (str): Type of model ('xgboost', 'lightgbm', 'rf', 'lr')
            test_size (float): Test set proportion
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.smote = None
        self.feature_cols = None
        self.metrics = {}
        
    def prepare_data(self, X, y):
        """
        Prepare data for modeling.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            
        Returns:
            tuple: Prepared datasets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.18, random_state=self.random_state, stratify=y_train
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_scaled, y_train)
        
        return (X_train_resampled, X_val_scaled, X_test_scaled,
                y_train_resampled, y_val, y_test)
    
    def create_model(self):
        """Create the model based on model_type."""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, eval_metric='logloss'
                )
            except ImportError:
                print("XGBoost not available, using Random Forest")
                self.model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=self.random_state
                )
                
        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1
                )
            except ImportError:
                print("LightGBM not available, using Random Forest")
                self.model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=self.random_state
                )
                
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
            )
            
        elif self.model_type == 'lr':
            self.model = LogisticRegression(
                random_state=self.random_state, max_iter=1000, class_weight='balanced'
            )
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            dict: Training metrics
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
    
    def tune_hyperparameters(self, X_train, y_train, param_grid=None):
        """
        Tune model hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid (dict): Parameter grid for tuning
        """
        if param_grid is None:
            if self.model_type in ['xgboost', 'lightgbm']:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1]
                }
            elif self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [8, 10],
                    'min_samples_split': [2, 5]
                }
            else:
                param_grid = {'C': [0.1, 1, 10]}
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.metrics['best_params'] = grid_search.best_params_
        self.metrics['best_cv_score'] = grid_search.best_score_
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Test metrics
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
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the model.
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    def save_model(self, filepath):
        """
        Save the model and related objects.
        
        Args:
            filepath (str): Base filepath for saving
        """
        joblib.dump(self.model, f"{filepath}_model.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.metrics, f"{filepath}_metrics.pkl")
        
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Base filepath for loading
        """
        self.model = joblib.load(f"{filepath}_model.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.metrics = joblib.load(f"{filepath}_metrics.pkl")


def train_model(X, y, model_type='xgboost'):
    """
    Convenience function to train a readmission model.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type (str): Type of model to train
        
    Returns:
        ReadmissionModel: Trained model
    """
    model = ReadmissionModel(model_type=model_type)
    
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X, y)
    
    model.train(X_train, y_train, X_val, y_val)
    
    return model, (X_test, y_test)
