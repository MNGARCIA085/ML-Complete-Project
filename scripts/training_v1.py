import hydra
from omegaconf import DictConfig
import mlflow
import json
import pandas as pd
from src.training.trainer import ModelTrainer
from src.data.preprocessor import Preprocessor
from src.models.model1 import build_compile_model_one
from src.models.baseline import build_compile_baseline

# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Data Preparation ===")
    # Initialize preprocessor
    preprocessor = Preprocessor(cfg.data)
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)
    train_ds = prep_results["train_ds"]
    val_ds = prep_results["val_ds"]
    input_shape = prep_results["input_shape"][0]
    scaler = prep_results["scaler"]
    encoder = prep_results["encoder"]
    features = prep_results["feature_columns"]

    # Set MLflow experiment
    mlflow.set_experiment(cfg.experiment_name)

    results_list = []

    models_to_train = ["baseline", "model1"]
    for model_name in models_to_train:
        print(f"=== Training model: {model_name} ===")
        # Compose the model config locally
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model

        # Build and compile the model
        model = MODEL_BUILDERS[model_name](model_cfg, input_dim=input_shape)

        # Init trainer
        trainer = ModelTrainer(cfg)

        with mlflow.start_run(run_name=model_name):
            # Train model
            results = trainer.train(model, train_ds, val_ds)

            # Log hyperparameters
            mlflow.log_params(results["hyperparameters"])
            mlflow.log_param("model_name", model_name)


            # Log metrics
            mlflow.log_metrics({
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_precision": results["val_precision"],
                "val_recall": results["val_recall"],
                "val_f1": results["val_f1"],
            })

            # Log model
            mlflow.tensorflow.log_model(model, artifact_path=cfg.logging.artifact_path)


            import tempfile
            import pickle
           

            # Log scaler
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                pickle.dump(scaler, f)
                f.flush()  # ensure content is written
                mlflow.log_artifact(f.name, artifact_path="preprocessing/scaler")

            # Log encoder
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                pickle.dump(encoder, f)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="preprocessing/encoder")

            # Log features as JSON
            mlflow.log_dict({"features": features}, "preprocessing/features.json")



            # Append for comparison DataFrame
            results_list.append({
                "model_name": model_name,
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_precision": results["val_precision"],
                "val_recall": results["val_recall"],
                "val_f1": results["val_f1"]
            })

    # Compare all models in a DataFrame
    df_results = pd.DataFrame(results_list)
    print("\n=== Model Comparison ===")
    print(df_results)

    # Optionally log the comparison table as JSON artifact
    mlflow.log_dict(df_results.to_dict(orient="records"), "model_comparison.json")

if __name__ == "__main__":
    main()
