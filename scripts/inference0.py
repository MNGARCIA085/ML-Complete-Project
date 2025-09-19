import mlflow
import mlflow.tensorflow
import pickle
import json
import pandas as pd
import numpy as np

def load_model_and_preprocessing(run_id: str):
    """
    Load a trained model and its preprocessing artifacts from MLflow.
    
    Args:
        run_id (str): MLflow run ID
    
    Returns:
        model: the trained TF/Keras model
        scaler: fitted scaler
        encoder: fitted encoder (if any, else None)
        features: list of feature column names
    """
    # Load model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.tensorflow.load_model(model_uri)
    
    # Load scaler
    scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessing/scaler")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load encoder
    encoder_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessing/encoder")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Load feature columns
    features_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessing/features.json")
    with open(features_path, "r") as f:
        features = json.load(f)["features"]
    
    return model, scaler, encoder, features


def predict_with_run(run_id: str, X_raw: np.ndarray):
    """
    Given raw input data, load the appropriate model & preprocessing from MLflow
    and return predictions.
    
    Args:
        run_id (str): MLflow run ID
        X_raw (np.ndarray): raw input data, shape (n_features,) or (n_samples, n_features)
    
    Returns:
        y_pred: predictions
    """
    model, scaler, encoder, features = load_model_and_preprocessing(run_id)
    
    # Convert to DataFrame with proper column names
    X_df = pd.DataFrame([X_raw], columns=features) if X_raw.ndim == 1 else pd.DataFrame(X_raw, columns=features)
    
    # Apply preprocessing
    X_scaled = scaler.transform(X_df)
    X_prepared = encoder.transform(X_scaled) if encoder is not None else X_scaled
    
    # Predict
    y_pred = model.predict(X_prepared)
    
    return y_pred


# ---------------- Example usage ----------------
"""
# Mock a new sample (shape 30,)
X_new_sample = np.random.rand(30)  # replace with actual patient data

# Replace with the run_id of the model you want to use
example_run_id = "your_mlflow_run_id_here"

y_pred = predict_with_run(example_run_id, X_new_sample)
print("Predicted output:", y_pred)
"""

