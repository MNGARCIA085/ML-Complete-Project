import mlflow.tensorflow
import pickle
from mlflow.tracking import MlflowClient

def load_best_model(experiment_name="Default", recall_threshold=0.8, artifact_model_path="model"):
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

    # Custom selection function
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

    # Load preprocessing artifacts
    scaler_path = f"mlruns/{experiment_id}/{run_id}/artifacts/preprocessing/scaler.pkl"
    encoder_path = f"mlruns/{experiment_id}/{run_id}/artifacts/preprocessing/encoder.pkl"

    scaler = pickle.load(open(scaler_path, "rb"))
    encoder = pickle.load(open(encoder_path, "rb"))

    return model, scaler, encoder



def main():
    model, scaler, encoder = load_best_model(
        experiment_name="Default",
        recall_threshold=0.8,
        artifact_model_path="model"
    )

    # Example inference
    import numpy as np


    print(model)
    print(scaler.mean_)
    print(encoder)

    """
    X_raw = np.array([[...]])  # raw features
    X_scaled = scaler.transform(X_raw)
    y_pred = model.predict(X_scaled)
    y_pred_decoded = encoder.inverse_transform(y_pred.argmax(axis=1))
    print("Prediction:", y_pred_decoded)
    """


if __name__==main():
    main()


"""
mlflow.set_tag("phase", "training")  # quick discard runs
mlflow.set_tag("phase", "tuning")    # hyperparameter optimization
mlflow.set_tag("phase", "best_overall")  # final production model
"""

"""
tag by phases then im gonna have to filter


import mlflow

# Set experiment
mlflow.set_experiment("MyProject")

# --- 1️⃣ Quick training runs (discard bad models) ---
for model_name in ["baseline", "model1"]:
    with mlflow.start_run(run_name=f"training_{model_name}") as run:
        mlflow.set_tag("phase", "training")
        # log params, metrics
        mlflow.log_params({...})
        mlflow.log_metrics({...})

# --- 2️⃣ Hyperparameter tuning runs ---
for model_name in ["baseline", "model1"]:
    with mlflow.start_run(run_name=f"tuning_{model_name}") as run:
        mlflow.set_tag("phase", "tuning")
        # log best params & metrics
        mlflow.log_params(best_hp.values)
        mlflow.log_metrics(best_metrics)

# --- 3️⃣ Comparison & best model ---
with mlflow.start_run(run_name="best_overall") as run:
    mlflow.set_tag("phase", "best_overall")
    # log best model params, metrics
    mlflow.log_params(best_hp.values)
    mlflow.log_metrics(best_metrics)
    mlflow.tensorflow.log_model(best_model, artifact_path="model")
    mlflow.log_artifacts("artifacts_preprocessing", artifact_path="preprocessing")
"""
