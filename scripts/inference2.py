import mlflow
import mlflow.tensorflow
import pickle
import json
import pandas as pd
import numpy as np

def get_latest_run_id(experiment_name: str, model_name: str): # i pout model cause it is the run name
    """
    Get the latest MLflow run_id for a given model in an experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Search for runs with the given model_name tag/parameter
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_name = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise ValueError(f"No runs found for model '{model_name}' in experiment '{experiment_name}'.")

    return runs.iloc[0]["run_id"]


import os

def load_model_and_preprocessing(run_id: str):
    """
    Load model and preprocessing artifacts from MLflow.
    """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.tensorflow.load_model(model_uri)
    
    # Scaler
    scaler_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessing/scaler")
    if os.path.isdir(scaler_dir):
        # pick the first file in the directory
        scaler_files = [f for f in os.listdir(scaler_dir) if f.endswith(".pkl")]
        if len(scaler_files) == 0:
            raise FileNotFoundError("No .pkl file found in scaler artifact directory")
        scaler_path = os.path.join(scaler_dir, scaler_files[0])
    else:
        scaler_path = scaler_dir
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Encoder
    encoder_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessing/encoder")
    if os.path.isdir(encoder_dir):
        encoder_files = [f for f in os.listdir(encoder_dir) if f.endswith(".pkl")]
        if len(encoder_files) == 0:
            raise FileNotFoundError("No .pkl file found in encoder artifact directory")
        encoder_path = os.path.join(encoder_dir, encoder_files[0])
    else:
        encoder_path = encoder_dir
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Features
    features_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessing/features.json")
    with open(features_path, "r") as f:
        features = json.load(f)["features"]
    
    return model, scaler, encoder, features


def predict(run_id: str, X_raw: np.ndarray):
    """
    Make predictions with a given run_id and raw input.
    """
    model, scaler, encoder, features = load_model_and_preprocessing(run_id)
    
    # Convert to DataFrame
    X_df = pd.DataFrame([X_raw], columns=features) if X_raw.ndim == 1 else pd.DataFrame(X_raw, columns=features)
    
    # Apply preprocessing (for now oly this, i already give mock data without unmamed:32....)
    X_scaled = scaler.transform(X_df)

    #X_prepared = encoder.transform(X_scaled) if encoder is not None else X_scaled
    
    y_pred = model.predict(X_scaled)
    return y_pred


# ---------------- Example usage ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow inference script.")
    parser.add_argument("--experiment", type=str, required=True, help="MLflow experiment name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name in MLflow params")
    parser.add_argument("--run_id", type=str, default=None, help="Optional specific MLflow run_id")
    parser.add_argument("--mock", action="store_true", help="Use mock data for testing")
    
    args = parser.parse_args()

    if args.run_id is None:
        print(f"No run_id provided. Selecting latest run of '{args.model_name}' in experiment '{args.experiment}'...")
        run_id = get_latest_run_id(args.experiment, args.model_name)
        print("Selected run_id:", run_id)
    else:
        run_id = args.run_id

    # Mock data (shape 30) if requested
    if args.mock:
        X_new = np.random.rand(30)
    else:
        # Replace with actual input
        X_new = np.random.rand(30)  # For demo; replace with real data

    y_pred = predict(run_id, X_new)
    print("Predicted output:", y_pred)


#python inference.py --experiment "BreastCancerExp" --model_name "baseline" --mock
#python inference.py --experiment "experiment_name" --model_name "baseline" --mock


#python -m scripts.inference2 --experiment "experiment_name" --model_name "model1" --mock
