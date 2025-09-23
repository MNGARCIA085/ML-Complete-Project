
import mlflow.tensorflow
import pickle
from mlflow.tracking import MlflowClient
from .selection import get_best_model
import json



def load_best_model(cfg, experiment_name, recall_threshold, artifact_model_path):
    """
    cfg-> corresponds to logging
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

    # Select best run (within this experiment)
    best_run = get_best_model(runs)
    run_id = best_run.info.run_id
    print(f"Selected best run_id: {run_id}")
    print(f"Metrics: {best_run.data.metrics}")

    # --- Load TensorFlow/Keras model ---
    model = mlflow.tensorflow.load_model(f"runs:/{run_id}/{artifact_model_path}")

    # --- Download preprocessing artifacts ---
    """
    problems with windows because of the slash
    scaler_path_local = client.download_artifacts(run_id, os.path.join(cfg.artifacts.preprocessing.base_dir, cfg.artifacts.preprocessing.scaler))
    encoder_path_local = client.download_artifacts(run_id, os.path.join(cfg.artifacts.preprocessing.base_dir, cfg.artifacts.preprocessing.encoder))
    """
    scaler_path_local = client.download_artifacts(
        run_id,
        f"{cfg.artifacts.preprocessing.base_dir}/{cfg.artifacts.preprocessing.scaler}"
    )
    encoder_path_local = client.download_artifacts(
        run_id,
        f"{cfg.artifacts.preprocessing.base_dir}/{cfg.artifacts.preprocessing.encoder}"
    )

    features_path_local = client.download_artifacts(
        run_id,
        f"{cfg.artifacts.preprocessing.base_dir}/{cfg.artifacts.preprocessing.features}"
    )

    # Load objects
    with open(scaler_path_local, "rb") as f:
        scaler = pickle.load(f)

    with open(encoder_path_local, "rb") as f:
        encoder = pickle.load(f)

    # Load JSON
    with open(features_path_local, "r") as f:
        features_data = json.load(f) # {'features': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',.....]}

    return model, scaler, encoder, features_data['features']
