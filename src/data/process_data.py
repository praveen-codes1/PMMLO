import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_dummy_data(output_path, num_records=1000):
    """Generate dummy sensor data for testing."""
    logger.info("Generating dummy sensor data...")
    
    # Create dummy data
    data = []
    for i in range(num_records):
        timestamp = datetime.now()
        machine_id = f"MACHINE_{i % 10}"
        sensor_data = {
            'timestamp': timestamp,
            'machine_id': machine_id,
            'temperature': np.random.normal(75, 5),
            'pressure': np.random.normal(100, 10),
            'vibration': np.random.normal(0.5, 0.1),
            'rpm': np.random.normal(1500, 100),
            'power_consumption': np.random.normal(1000, 50),
            'oil_level': np.random.normal(80, 5),
            'maintenance_required': np.random.choice([0, 1], p=[0.9, 0.1])
        }
        data.append(sensor_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save as parquet
    df.to_parquet(output_path)
    logger.info(f"Generated {num_records} dummy records")

def process_data():
    """Main function to process the data."""
    try:
        # Define paths
        data_dir = os.path.join(project_root, "data")
        raw_data_path = os.path.join(data_dir, "raw", "sensor_data.parquet")
        processed_data_path = os.path.join(data_dir, "processed", "processed_data.parquet")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

        # Check if raw data exists, if not generate dummy data
        if not os.path.exists(raw_data_path):
            logger.info("No raw data found. Generating dummy data...")
            generate_dummy_data(raw_data_path)
        
        # Read raw data
        logger.info("Reading raw data...")
        df = pd.read_parquet(raw_data_path)
        
        # Basic data cleaning
        logger.info("Cleaning data...")
        df_cleaned = df.drop_duplicates()
        df_cleaned = df_cleaned.dropna(subset=['temperature', 'pressure', 'vibration'])
        
        # Add derived features
        logger.info("Adding derived features...")
        df_processed = df_cleaned.copy()
        df_processed['temperature_pressure_ratio'] = df_processed['temperature'] / df_processed['pressure']
        df_processed['vibration_rpm_ratio'] = df_processed['vibration'] * df_processed['rpm']
        
        # Save processed data
        logger.info("Saving processed data...")
        df_processed.to_parquet(processed_data_path)
        
        logger.info("Data processing completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return False

if __name__ == "__main__":
    process_data() 