from setuptools import setup, find_packages

setup(
    name="pmmlo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "plotly",
        "prometheus-client"
    ],
    python_requires=">=3.8",
) 