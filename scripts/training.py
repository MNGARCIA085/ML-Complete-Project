import hydra
from omegaconf import DictConfig
import mlflow
import pandas as pd
import tempfile
import pickle
from src.training.trainer import ModelTrainer
from src.data.preprocessor import Preprocessor
from src.models.model1 import build_compile_model_one
from src.models.baseline import build_compile_baseline
from src.utils.plotting import plot_history_curve, plot_confusion_matrix, plot_comparison



# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    print("=== Data Preparation ===")

    # Initialize preprocessor with cfg.data
    preprocessor = Preprocessor(cfg.data)

    # train/val
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)

    # keys
    keys = ["train_ds", "val_ds", "scaler", "encoder", "feature_columns", "input_shape"]
    train_ds, val_ds, scaler, encoder, features, input_shape_raw = (prep_results[k] for k in keys)
    input_shape = input_shape_raw[0]  # keep the first element as before

    # metrics to log
    metrics = ["val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"]


    # Set MLflow experiment
    mlflow.set_experiment(cfg.logging.experiment_name)

    results_list = []


    models_to_train = ["baseline", "model1"]
    for model_name in models_to_train:
        
        print(f"=== Training model: {model_name} ===")
        
        # model
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model # override with the apporpiate file (e.g. baseline.yaml or model1.yaml...)
        model = MODEL_BUILDERS[model_name](model_cfg, input_dim=input_shape) # model already build and compiled

        # trainer
        trainer = ModelTrainer(cfg.training)

        # run trainer and logging
        with mlflow.start_run(run_name=model_name):
            # Train model
            results = trainer.train(model, train_ds, val_ds)
            history = results["history"].history

            # Log hyperparameters
            mlflow.log_params(results["hyperparameters"])
            mlflow.log_param("model_name", model_name)


            # Log metrics
            mlflow.log_metrics({m: results[m] for m in metrics})

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
            loss_plot_path = plot_history_curve(
                history, "loss", "val_loss", model_name, "Loss", "Loss", "loss.png"
            )
            acc_plot_path = plot_history_curve(
                history, "accuracy", "val_accuracy", model_name, "Accuracy", "Accuracy", "accuracy.png"
            )

            # ----- Confusion matrix -----
            cm_plot_path = plot_confusion_matrix(model, val_ds, model_name, "confusion_matrix.png")

            # ----- Logging outside -----
            mlflow.log_artifact(loss_plot_path, artifact_path="plots")
            mlflow.log_artifact(acc_plot_path, artifact_path="plots")
            mlflow.log_artifact(cm_plot_path, artifact_path="plots")


            # Store results for comparison

            results_list.append({
                "model_name": model_name,
                "history": results["history"],
                **{m: results[m] for m in metrics}
            })


    # ----- Comparison run -----
    with mlflow.start_run(run_name="all_models_comparison"):

        # ----- Comparison plots -----
        comp_loss_path = plot_comparison(
            results_list, metric="loss", ylabel="Loss",
            title="Validation Loss Comparison", filename="comparison_val_loss.png"
        )

        comp_acc_path = plot_comparison(
            results_list, metric="accuracy", ylabel="Accuracy",
            title="Validation Accuracy Comparison", filename="comparison_val_accuracy.png"
        )

        # ----- Logging outside -----
        mlflow.log_artifact(comp_loss_path, artifact_path="plots")
        mlflow.log_artifact(comp_acc_path, artifact_path="plots")


        # Log metrics comparison table
        df_results = pd.DataFrame([
            {
                "model_name": r["model_name"],
                **{m: r[m] for m in metrics}
            }
            for r in results_list
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