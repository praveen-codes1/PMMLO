import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Set page config
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri("file://" + os.path.join(project_root, "mlruns"))
    return mlflow.sklearn.load_model("models:/predictive_maintenance/Production")

@st.cache_data
def load_data():
    data_path = os.path.join(project_root, "data", "processed")
    return pd.read_parquet(data_path)

def main():
    st.title("🔧 Predictive Maintenance Dashboard")
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.info("""
    This dashboard shows real-time predictions and model performance metrics
    for the predictive maintenance system.
    """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sensor Data Distribution")
        fig = px.box(data, y=['temperature', 'pressure', 'vibration', 'rpm'],
                    title="Sensor Readings Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Maintenance Predictions")
        maintenance_counts = data['maintenance_required'].value_counts()
        fig = px.pie(values=maintenance_counts.values, 
                    names=['No Maintenance', 'Maintenance Required'],
                    title="Maintenance Predictions Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='feature', y='importance',
                 title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Real-time predictions
    st.subheader("Make a Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature", 50.0, 100.0, 75.0)
        pressure = st.slider("Pressure", 50.0, 150.0, 100.0)
        vibration = st.slider("Vibration", 0.1, 1.0, 0.5)
    
    with col2:
        rpm = st.slider("RPM", 1000.0, 2000.0, 1500.0)
        power_consumption = st.slider("Power Consumption", 800.0, 1200.0, 1000.0)
        oil_level = st.slider("Oil Level", 50.0, 100.0, 80.0)
    
    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'temperature': temperature,
            'pressure': pressure,
            'vibration': vibration,
            'rpm': rpm,
            'power_consumption': power_consumption,
            'oil_level': oil_level,
            'temperature_pressure_ratio': temperature / pressure,
            'vibration_rpm_ratio': vibration * rpm
        }])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.success(f"""
        Prediction: {'Maintenance Required' if prediction else 'No Maintenance Required'}
        Confidence: {probability:.2%}
        """)

if __name__ == "__main__":
    main() 