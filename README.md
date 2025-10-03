## ML Complete Project

This repository contains a machine learning project with scripts for data preprocessing, training, evaluation, and inference. It integrates **MLflow** for experiment tracking and artifact management.

---

### Setup

#### 1. Create a virtual environment and install dependencies

```bash
# Create venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


#### 2. Run the main pipeline

```bash
python -m scripts.pipeline
```

#### 3. Run tests

```bash
pytest
```

### Docker usage

#### 1. Build the container

```bash
docker build -f Dockerfile.ml -t ml_env:latest .
```

#### 2. Run the container

```bash
docker run -it -v $(pwd):/app ml_env:latest
```

#### 3. Run MLFLow inside Docker

```bash
docker run -it -p 5000:5000 -v $(pwd):/app ml_env:latest mlflow ui --host 0.0.0.0 --port 5000
```


