import os
import sys
import logging
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Predictive Maintenance API")

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'prediction_total',
    'Total number of predictions made'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction request'
)

# Load model
mlflow.set_tracking_uri("file://" + os.path.join(project_root, "mlruns"))
model = mlflow.sklearn.load_model("models:/predictive_maintenance/Production")

class SensorData(BaseModel):
    temperature: float
    pressure: float
    vibration: float
    rpm: float
    power_consumption: float
    oil_level: float

@app.get("/")
async def root():
    return {"message": "Predictive Maintenance API"}

@app.post("/predict")
async def predict(data: SensorData):
    try:
        start_time = time.time()
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'temperature': data.temperature,
            'pressure': data.pressure,
            'vibration': data.vibration,
            'rpm': data.rpm,
            'power_consumption': data.power_consumption,
            'oil_level': data.oil_level,
            'temperature_pressure_ratio': data.temperature / data.pressure,
            'vibration_rpm_ratio': data.vibration * data.rpm
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return {
            "maintenance_required": bool(prediction),
            "probability": float(probability)
        }

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 