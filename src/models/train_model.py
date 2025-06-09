import os
import sys
import logging
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load processed data."""
    data_path = os.path.join(project_root, "data", "processed", "processed_data.parquet")
    return pd.read_parquet(data_path)

def initialize_mlflow():
    """Initialize MLflow tracking."""
    try:
        # Set MLflow tracking URI to file-based URI
        tracking_uri = f"file://{os.path.join(project_root, 'mlruns')}"
        os.makedirs(os.path.join(project_root, 'mlruns'), exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        experiment_name = "predictive_maintenance"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        return True
    except Exception as e:
        logger.error(f"Error initializing MLflow: {str(e)}")
        return False

def train_model():
    """Train the predictive maintenance model."""
    try:
        # Initialize MLflow
        if not initialize_mlflow():
            return False

        # Load data
        logger.info("Loading processed data...")
        data = load_data()

        # Prepare features and target
        features = ['temperature', 'pressure', 'vibration', 'rpm', 
                   'power_consumption', 'oil_level', 
                   'temperature_pressure_ratio', 'vibration_rpm_ratio']
        target = 'maintenance_required'

        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Start MLflow run
        with mlflow.start_run():
            # Train model
            logger.info("Training Random Forest model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            })
            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

            logger.info("Model training completed successfully")
            logger.info(f"Model metrics: {metrics}")

        return True

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 