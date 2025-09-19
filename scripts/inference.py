from mlflow.tracking import MlflowClient
import mlflow.tensorflow
import pickle
import os


import numpy as np


def load_best_model(cfg, experiment_name="Test", recall_threshold=0.8, artifact_model_path="model"):
    """
    Returns: model, scaler, encoder ready for inference
    Selects best model according to:
      - recall >= recall_threshold
      - among those, highest F1 score
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    # Get all best_overall runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName = 'best_overall'"
    )

    if not runs:
        raise ValueError("No 'best_overall' runs found in the experiment.")

    # Select best run
    candidates = [r for r in runs if r.data.metrics.get("recall", 0) >= recall_threshold]
    if candidates:
        best_run = max(candidates, key=lambda r: r.data.metrics.get("f1_score", 0))
    else:
        best_run = max(runs, key=lambda r: r.data.metrics.get("recall", 0))

    run_id = best_run.info.run_id
    print(f"Selected best run_id: {run_id}")
    print(f"Metrics: {best_run.data.metrics}")

    # --- Load TensorFlow/Keras model ---
    model = mlflow.tensorflow.load_model(f"runs:/{run_id}/{artifact_model_path}")

    # --- Download preprocessing artifacts ---
    scaler_path_local = client.download_artifacts(run_id, os.path.join(cfg.artifacts.preprocessing.base_dir, cfg.artifacts.preprocessing.scaler))
    encoder_path_local = client.download_artifacts(run_id, os.path.join(cfg.artifacts.preprocessing.base_dir, cfg.artifacts.preprocessing.encoder))

    # Load objects
    with open(scaler_path_local, "rb") as f:
        scaler = pickle.load(f)

    with open(encoder_path_local, "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder




import pandas as pd


def predict_with_run(cfg, X_raw: np.ndarray):
    """
    Given raw input data, load the appropriate model & preprocessing from MLflow
    and return predictions.
    
    Args:
        run_id (str): MLflow run ID
        X_raw (np.ndarray): raw input data, shape (n_features,) or (n_samples, n_features)
    
    Returns:
        y_pred: predictions
    """
    model, scaler, encoder  = load_best_model(cfg) # features
    
    # Convert to DataFrame with proper column names
    #X_df = pd.DataFrame([X_raw], columns=features) if X_raw.ndim == 1 else pd.DataFrame(X_raw, columns=features)
    X_df = pd.DataFrame([X_raw])
    
    # Apply preprocessing
    X_scaled = scaler.transform(X_df)
    #X_prepared = encoder.transform(X_scaled) if encoder is not None else X_scaled
    
    # Predict
    y_pred = model.predict(X_scaled)
    
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



import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('sdfdsfdsfdsfsd')
    X_new_sample = np.random.rand(30)  # replace with actual patient data

    print(X_new_sample)

    y_pred = predict_with_run(cfg.logging, X_new_sample)
    print(y_pred)


if __name__==main():
    main()