"""
Tests for Hospital Readmission Predictor.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataCleaning:
    """Test data cleaning functions."""
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        from src.data_cleaning import DataCleaner
        
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['a', 'b', 'c', 'd'],
            'readmitted': ['<30', '>30', 'NO', '<30']
        })
        
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(df)
        
        assert df_clean is not None
        assert len(df_clean) > 0


class TestModel:
    """Test model functions."""
    
    def test_model_creation(self):
        """Test model can be created."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        assert model is not None
    
    def test_model_fit(self):
        """Test model can be fitted."""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert model is not None
    
    def test_model_predict(self):
        """Test model prediction."""
        from sklearn.ensemble import RandomForestClassifier
        
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(10, 5)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


class TestVisualization:
    """Test visualization functions."""
    
    def test_visualizer_init(self):
        """Test visualizer initialization."""
        from src.visualization import ReadmissionVisualizer
        
        viz = ReadmissionVisualizer()
        assert viz is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
