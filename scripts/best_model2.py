import mlflow.tensorflow
import mlflow
import pickle
import json
from mlflow.tracking import MlflowClient


import os


def load_best_model(experiment_name="Default", recall_threshold=0.8, artifact_model_path="model", cfg=None):
    """
    Returns: model, scaler, encoder, features ready for inference.

    Selects the best model among all 'best_overall' runs according to:
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

    # Load TensorFlow/Keras model
    model = mlflow.tensorflow.load_model(f"runs:/{run_id}/{artifact_model_path}")

    # Scaler
    scaler_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=cfg.logging.preprocessing.scaler if cfg else "preprocessing/scaler"
    )
    scaler_path = os.path.join(scaler_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Encoder
    encoder_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=cfg.logging.preprocessing.encoder if cfg else "preprocessing/encoder"
    )
    encoder_path = os.path.join(encoder_dir, "encoder.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    # Features (this is a file, so no join needed)
    features_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=cfg.logging.preprocessing.features if cfg else "preprocessing/features.json"
    )
    with open(features_path, "r") as f:
        features = json.load(f)["features"]

    return model, scaler, encoder, features


def main():
    # Optional: pass config if you want to use paths from YAML
    model, scaler, encoder, features = load_best_model(
        experiment_name="Default",
        recall_threshold=0.8,
        artifact_model_path="model"
    )

    # Example inference
    import numpy as np
    print(model)
    print(scaler.mean_)
    print(encoder)
    print(features)


if __name__ == "__main__":
    main()
