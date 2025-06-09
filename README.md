# Predictive Maintenance MLOps Pipeline

An open-source MLOps pipeline for predictive maintenance, featuring data processing with Apache Spark, model tracking with MLflow, model serving via Dockerized Flask API, and monitoring with Streamlit.

## Prerequisites

- Python 3.8+
- Docker Desktop
- Apache Spark
- Java Runtime Environment
- Git

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/predictive_maint_mlops.git
cd predictive_maint_mlops
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start MLflow tracking server:
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 --port 5000
```

## Usage

### Data Processing
```bash
spark-submit process_data.py
```

### Model Training
```bash
python train_model.py
```

### Model Serving
```bash
docker build -t predictive-maint-api .
docker run -p 8080:8080 predictive-maint-api
```

### Monitoring Dashboard
```bash
streamlit run dashboard.py
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:
1. On each push to main, tests are run
2. If tests pass, a new Docker image is built and pushed to Docker Hub
3. The image is tagged with the commit SHA

## Project Structure

- `data_raw/`: Raw input data
- `data_processed/`: Processed data files
- `features/`: Feature engineering scripts
- `tests/`: Unit tests
- `mlflow-artifacts/`: MLflow model artifacts
- `process_data.py`: Spark data processing script
- `train_model.py`: Model training script
- `serve_model.py`: Flask API for model serving
- `dashboard.py`: Streamlit monitoring dashboard