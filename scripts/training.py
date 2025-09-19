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





# logging_utils .> ml_logger
def log_experiment_results(results, model_name, model, scaler, encoder, features,
                           metrics, cfg, loss_plot_path, acc_plot_path, cm_plot_path):
    
    # Log model name
    mlflow.log_param("model_name", model_name)

    # Log hyperparameters
    mlflow.log_params(results["hyperparameters"])
    
    # Log metrics
    mlflow.log_metrics({m: results[m] for m in metrics})

    if cfg.save_artifacts:
        # Log model
        mlflow.tensorflow.log_model(model, artifact_path=cfg.artifact_path)

        # Log preprocessing artifacts
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            pickle.dump(scaler, f)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path=cfg.preprocessing.scaler)

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            pickle.dump(encoder, f)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path=cfg.preprocessing.encoder)

        mlflow.log_dict({"features": features}, cfg.preprocessing.features)

    # Always log plots (these are lightweight and useful for debugging)
    mlflow.log_artifact(loss_plot_path, artifact_path=cfg.plots)
    mlflow.log_artifact(acc_plot_path, artifact_path=cfg.plots)
    mlflow.log_artifact(cm_plot_path, artifact_path=cfg.plots)



# log comparisson
def log_model_comparison(results_list, metrics, plot_comparison, cfg):
    # Comparison plots
    comp_loss_path = plot_comparison(
        results_list, metric="loss", ylabel="Loss",
        title="Validation Loss Comparison", filename="comparison_val_loss.png"
    )

    comp_acc_path = plot_comparison(
        results_list, metric="accuracy", ylabel="Accuracy",
        title="Validation Accuracy Comparison", filename="comparison_val_accuracy.png"
    )

    # Log plots
    mlflow.log_artifact(comp_loss_path, artifact_path=cfg.plots)
    mlflow.log_artifact(comp_acc_path, artifact_path=cfg.plots)

    # Metrics comparison table
    df_results = pd.DataFrame([
        {
            "model_name": r["model_name"],
            **{m: r[m] for m in metrics}
        }
        for r in results_list
    ])

    print("\n=== Model Comparison ===")
    print(df_results)

    mlflow.log_dict(df_results.to_dict(orient="records"), cfg.comparison_table)









@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    print("=== Data Preparation ===")

    # Initialize preprocessor
    preprocessor = Preprocessor(cfg)

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
        model = MODEL_BUILDERS[model_name](model_cfg, cfg.training, input_dim=input_shape) # model already build and compiled

        # trainer
        trainer = ModelTrainer(cfg.training)

        # run trainer and logging
        with mlflow.start_run(run_name=model_name):
            # Train model
            results = trainer.train(model, train_ds, val_ds)
            history = results["history"].history

            # ----- History plots -----
            loss_plot_path = plot_history_curve(
                history, "loss", "val_loss", model_name, "Loss", "Loss", "loss.png"
            )
            acc_plot_path = plot_history_curve(
                history, "accuracy", "val_accuracy", model_name, "Accuracy", "Accuracy", "accuracy.png"
            )

            # ----- Confusion matrix -----
            cm_plot_path = plot_confusion_matrix(model, val_ds, model_name, "confusion_matrix.png")


            # logging
            log_experiment_results(
                results=results,
                model_name=model_name,
                model=model,
                scaler=scaler,
                encoder=encoder,
                features=features,
                metrics=metrics,
                cfg=cfg.logging,
                loss_plot_path=loss_plot_path,
                acc_plot_path=acc_plot_path,
                cm_plot_path=cm_plot_path,
            )


            # Store results for comparison
            results_list.append({
                "model_name": model_name,
                "history": results["history"],
                **{m: results[m] for m in metrics}
            })


    # ----- Comparison run -----
    with mlflow.start_run(run_name="all_models_comparison"):
        # log comparisson
        log_model_comparison(
            results_list=results_list,
            metrics=metrics,
            plot_comparison=plot_comparison,
            cfg=cfg.logging
        )

if __name__ == "__main__":
    main()



"""

python -m scripts.training training.epochs=3


use particular file

global_cfg = hydra.compose(config_name="config/config.yaml")
model_cfg = hydra.compose(config_name="config/model/baseline_tuner.yaml").model

"""