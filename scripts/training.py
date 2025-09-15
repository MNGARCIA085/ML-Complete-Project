import hydra
from omegaconf import DictConfig
import mlflow
import json
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    preprocessor = Preprocessor(cfg.data)
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)
    train_ds = prep_results["train_ds"]
    val_ds = prep_results["val_ds"]
    input_shape = prep_results["input_shape"][0]
    scaler = prep_results["scaler"]
    encoder = prep_results["encoder"]
    features = prep_results["feature_columns"]

    # Set MLflow experiment
    mlflow.set_experiment(cfg.logging.experiment_name)

    results_list = []

    models_to_train = ["baseline", "model1"]
    for model_name in models_to_train:
        print(f"=== Training model: {model_name} ===")
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model
        model = MODEL_BUILDERS[model_name](model_cfg, input_dim=input_shape)
        trainer = ModelTrainer(cfg)

        with mlflow.start_run(run_name=model_name):
            # Train model
            results = trainer.train(model, train_ds, val_ds)
            history = results["history"].history

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

            # Log preprocessing artifacts
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                pickle.dump(scaler, f)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="preprocessing/scaler")

            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                pickle.dump(encoder, f)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path="preprocessing/encoder")

            mlflow.log_dict({"features": features}, "preprocessing/features.json")

            # ----- History plots -----
            # Loss
            plt.figure()
            plt.plot(history["loss"], label="train_loss")
            plt.plot(history["val_loss"], label="val_loss")
            plt.title(f"{model_name} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            loss_plot_path = f"{model_name}_loss.png"
            plt.savefig(loss_plot_path)
            plt.close()
            mlflow.log_artifact(loss_plot_path, artifact_path="plots")

            # Accuracy
            plt.figure()
            plt.plot(history["accuracy"], label="train_acc")
            plt.plot(history["val_accuracy"], label="val_acc")
            plt.title(f"{model_name} Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            acc_plot_path = f"{model_name}_accuracy.png"
            plt.savefig(acc_plot_path)
            plt.close()
            mlflow.log_artifact(acc_plot_path, artifact_path="plots")

            # ----- Confusion matrix -----
            # Extract labels from val_ds
            y_true = np.concatenate([y for _, y in val_ds], axis=0)
            y_pred = np.argmax(model.predict(val_ds), axis=1)
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{model_name} Confusion Matrix")
            plt.tight_layout()
            cm_path = f"{model_name}_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path, artifact_path="plots")

            """
            cleanr saving
            import time
            cm_path = f"{model_name}_confusion_matrix_{int(time.time())}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            or cm_path = f"{model_name}_confusion_matrix_epoch{epoch}.png"
            """

            # Store results for comparison
            results_list.append({
                "model_name": model_name,
                "history": results["history"],
                "val_loss": results["val_loss"],
                "val_accuracy": results["val_accuracy"],
                "val_precision": results["val_precision"],
                "val_recall": results["val_recall"],
                "val_f1": results["val_f1"]
            })

    # ----- Comparison run -----
    with mlflow.start_run(run_name="all_models_comparison"):
        # Loss comparison
        plt.figure()
        for r in results_list:
            h = r["history"].history
            plt.plot(h["val_loss"], label=f"{r['model_name']} val_loss")
        plt.title("Validation Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        comp_loss_path = "comparison_val_loss.png"
        plt.savefig(comp_loss_path)
        plt.close()
        mlflow.log_artifact(comp_loss_path, artifact_path="plots")

        # Accuracy comparison
        plt.figure()
        for r in results_list:
            h = r["history"].history
            plt.plot(h["val_accuracy"], label=f"{r['model_name']} val_acc")
        plt.title("Validation Accuracy Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        comp_acc_path = "comparison_val_accuracy.png"
        plt.savefig(comp_acc_path)
        plt.close()
        mlflow.log_artifact(comp_acc_path, artifact_path="plots")

        # Log metrics comparison table
        df_results = pd.DataFrame([
            {
                "model_name": r["model_name"],
                "val_loss": r["val_loss"],
                "val_accuracy": r["val_accuracy"],
                "val_precision": r["val_precision"],
                "val_recall": r["val_recall"],
                "val_f1": r["val_f1"]
            } for r in results_list
        ])
        print("\n=== Model Comparison ===")
        print(df_results)
        mlflow.log_dict(df_results.to_dict(orient="records"), "model_comparison.json")


if __name__ == "__main__":
    main()



"""

use particular file

global_cfg = hydra.compose(config_name="config/config.yaml")

model_cfg = hydra.compose(config_name="config/model/baseline_tuner.yaml").model

tuner = ModelTuner(build_fn=build_compile_model_one_tuner,
                   tuning_cfg=model_cfg,
                   input_dim=global_cfg.input_dim,
                   seed=global_cfg.seed)

"""