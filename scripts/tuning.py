import hydra
from omegaconf import DictConfig
import mlflow
import pickle
import os

from src.tuning.tuner import ModelTuner
from src.data.preprocessor import Preprocessor
from src.models.model1 import build_compile_model_one
from src.models.baseline import build_compile_baseline

MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Data Preparation ===")
    preprocessor = Preprocessor(cfg.data)
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)

    keys = ["train_ds", "val_ds", "scaler", "encoder", "feature_columns", "input_shape"]
    train_ds, val_ds, scaler, encoder, features, input_shape_raw = (prep_results[k] for k in keys)
    input_dim = input_shape_raw[0]

    print("=== Models Tuning ===")
    models_to_tune = ["baseline", "model1"]
    all_results = []

    # 1️⃣ Run for each model: log hyperparams + metrics
    for model_name in models_to_tune:
        print(f"=== Training model: {model_name} ===")
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model
        build_fn = MODEL_BUILDERS[model_name]

        tuner = ModelTuner(cfg.tuning, cfg.training, model_cfg, build_fn, input_dim)
        best_model, best_hp, val_metrics = tuner.run(train_ds, val_ds)

        # Track result for comparison
        all_results.append({
            "name": model_name,
            "model": best_model,
            "hp": best_hp.values,
            "metrics": val_metrics
        })

        # MLflow run per model
        with mlflow.start_run(run_name=f"{model_name}_run"):
            mlflow.log_params(best_hp.values)
            mlflow.log_metrics(val_metrics)

        print(model_name)
        print("Best hyperparameters:", best_hp.values)
        print("Validation metrics:", val_metrics)

    # 2️⃣ Comparison run: log all metrics, pick best
    with mlflow.start_run(run_name="comparison") as comp_run:
        for res in all_results:
            prefix = res["name"]
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in res["metrics"].items()})

        # Pick best model using threshold + f1
        threshold = 0.80 # later from config
        candidates = [res for res in all_results if res["metrics"]["recall"] >= threshold]
        if candidates:
            best = max(candidates, key=lambda r: r["metrics"]["f1_score"])
        else:
            best = max(all_results, key=lambda r: r["metrics"]["recall"])

        mlflow.log_params({"best_model_name": best["name"]})
        mlflow.log_metrics({
            "best_model_recall": best["metrics"]["recall"],
            "best_model_f1": best["metrics"]["f1_score"]
        })

        comparison_run_id = comp_run.info.run_id  # save if needed

    # 3️⃣ Final run: store best model + preprocessing + metrics
    with mlflow.start_run(run_name="best_overall") as best_run:
        mlflow.log_params(best["hp"])
        mlflow.log_metrics(best["metrics"])

        # Log extra info
        mlflow.log_param("model_name", best["name"])
        mlflow.log_param("epochs", cfg.tuning.epochs)  # or best_hp["epochs"] if tuned

        # Save TensorFlow/Keras model
        mlflow.tensorflow.log_model(
            model=best["model"],
            artifact_path=cfg.logging.artifact_path  # e.g. "model"
        )

        # Save preprocessing artifacts
        artifact_dir = f"artifacts_{best['name']}"
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(artifact_dir, "encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

        # columns!!!!!!
        mlflow.log_dict({"features": features}, "preprocessing/features.json")

        mlflow.log_artifacts(artifact_dir, artifact_path="preprocessing")



    print(f"Overall best model: {best['name']}")
    print("Best hyperparameters:", best["hp"])
    print("Best validation metrics:", best["metrics"])
    print("Final best model run_id:", best_run.info.run_id)


if __name__ == "__main__":
    main()



"""
global_cfg = hydra.compose(config_name="config/config.yaml")
model_cfg = hydra.compose(config_name="config/model/model1.yaml").model
"""

"""
To get all runs
When you start a run:

with mlflow.start_run(run_name="comparison") as run:
    run_id = run.info.run_id
    print("Comparison run_id:", run_id)


Save that run_id somewhere (config, JSON, DB, etc.) for later inference.
"""




"""
best among all runs
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("Default")  # or your experiment name

# Search runs sorted by recall, then f1
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.recall DESC", "metrics.f1_score DESC"]
)

best_run = runs[0]
print("Best run:", best_run.info.run_id, best_run.data.metrics)
"""




"""
import mlflow.tensorflow
import pickle

# Example: run_id from MLflow UI or returned by start_run
run_id = "<comparison_run_id>"  
artifact_path = "model"  # same as cfg.logging.artifact_path

# Load the model
model = mlflow.tensorflow.load_model(f"runs:/{run_id}/{artifact_path}")

# Load preprocessing artifacts (scaler, encoder)
scaler = pickle.load(open(f"mlruns/{run_id}/artifacts/preprocessing/scaler.pkl", "rb"))
encoder = pickle.load(open(f"mlruns/{run_id}/artifacts/preprocessing/encoder.pkl", "rb"))


import numpy as np

# Example input (raw feature row)
X_raw = np.array([[...]])   # shape must match training features

# Apply preprocessing
X_scaled = scaler.transform(X_raw)

# Model prediction
pred = model.predict(X_scaled)

# Decode output if classification
y_pred = encoder.inverse_transform(pred.argmax(axis=1))
print("Prediction:", y_pred)


"""