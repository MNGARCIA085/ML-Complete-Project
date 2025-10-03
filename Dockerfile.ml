# Use Debian base from Google's Docker Hub mirror
# Dockerfile.ml
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what is needed for training
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY config/ config
COPY tests/ tests
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY notebooks/ notebooks/

# Environment variables
ENV PYTHONUNBUFFERED=1

CMD ["bash"]




#docker build -f Dockerfile.ml -t ml_env:latest .
#docker run -it -v $(pwd):/app ml_env:latest
#docker run -it -p 5000:5000 -v $(pwd):/app ml_env:latest; mlflow ui --host 0.0.0.0 --port 5000
