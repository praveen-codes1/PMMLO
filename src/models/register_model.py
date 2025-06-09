import os
import mlflow
from mlflow.tracking import MlflowClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
mlruns_path = os.path.join(project_root, "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

EXPERIMENT_NAME = "predictive_maintenance"
MODEL_NAME = "predictive_maintenance"

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")

# Get latest run
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
if not runs:
    raise Exception("No runs found in experiment.")
run = runs[0]
model_uri = f"runs:/{run.info.run_id}/model"

# Register model
result = mlflow.register_model(model_uri, MODEL_NAME)

# Transition to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Registered model {MODEL_NAME} version {result.version} to Production.") 