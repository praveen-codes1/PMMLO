import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.models.train_model import train_model
from src.api.app import predict, SensorData

def test_model_training():
    """Test if model training completes successfully."""
    assert train_model() == True

def test_prediction_api():
    """Test the prediction API endpoint."""
    # Create sample sensor data
    sensor_data = SensorData(
        temperature=75.0,
        pressure=100.0,
        vibration=0.5,
        rpm=1500.0,
        power_consumption=1000.0,
        oil_level=80.0
    )
    
    # Make prediction
    result = predict(sensor_data)
    
    # Check response format
    assert isinstance(result, dict)
    assert 'maintenance_required' in result
    assert 'probability' in result
    assert isinstance(result['maintenance_required'], bool)
    assert isinstance(result['probability'], float)
    assert 0 <= result['probability'] <= 1

def test_feature_importance():
    """Test if feature importance is calculated correctly."""
    # Create sample data
    X = pd.DataFrame({
        'temperature': np.random.normal(75, 5, 100),
        'pressure': np.random.normal(100, 10, 100),
        'vibration': np.random.normal(0.5, 0.1, 100),
        'rpm': np.random.normal(1500, 100, 100),
        'power_consumption': np.random.normal(1000, 50, 100),
        'oil_level': np.random.normal(80, 5, 100),
        'temperature_pressure_ratio': np.random.normal(0.75, 0.1, 100),
        'vibration_rpm_ratio': np.random.normal(0.75, 0.1, 100)
    })
    y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Check feature importance
    assert len(model.feature_importances_) == len(X.columns)
    assert all(0 <= imp <= 1 for imp in model.feature_importances_)
    assert abs(sum(model.feature_importances_) - 1.0) < 1e-10

if __name__ == "__main__":
    pytest.main([__file__]) 