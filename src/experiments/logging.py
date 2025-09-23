import mlflow
import pandas as pd
import tempfile
import pickle
import os



# logging_utils .> ml_logger
def log_experiment_results(results, model_name, model, scaler, encoder, features,
                           metrics, loss_plot_path, acc_plot_path, cm_plot_path, global_cfg):


    with mlflow.start_run(run_name=model_name):
    
        # Log model name
        mlflow.log_param("model_name", model_name)

        # relaeted to the ds
        mlflow.log_param("val_split", global_cfg.data.preprocessing.val_size)

        # Log hyperparameters
        mlflow.log_params(results["hyperparameters"])
        
        # Log metrics
        mlflow.log_metrics({m: results[m] for m in metrics})

        if global_cfg.logging.save_artifacts:
            # Log model
            mlflow.tensorflow.log_model(model, artifact_path=global_cfg.logging.artifacts.model) # artifact_path

            # Log preprocessing artifacts
            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                pickle.dump(scaler, f)
                f.flush()
                mlflow.log_artifact(f.name, 
                    artifact_path=f"{global_cfg.logging.artifacts.preprocessing.base_dir}/{global_cfg.logging.artifacts.preprocessing.scaler}")

            with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
                pickle.dump(encoder, f)
                f.flush()
                mlflow.log_artifact(f.name, artifact_path=f"{global_cfg.logging.artifacts.preprocessing.base_dir}/{global_cfg.logging.artifacts.preprocessing.encoder}")

            mlflow.log_dict({"features": features}, f"{global_cfg.logging.artifacts.preprocessing.base_dir}/{global_cfg.logging.artifacts.preprocessing.features}")

        # Always log plots (these are lightweight and useful for debugging)
        mlflow.log_artifact(loss_plot_path, artifact_path=global_cfg.logging.plots)
        mlflow.log_artifact(acc_plot_path, artifact_path=global_cfg.logging.plots)
        mlflow.log_artifact(cm_plot_path, artifact_path=global_cfg.logging.plots)



# log comparisson
def log_model_comparison(results_list, metrics, plot_comparison, cfg):

    with mlflow.start_run(run_name="all_models_comparison"):
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


# log best model
def log_best_model(best, cfg, scaler, encoder, features):
    with mlflow.start_run(run_name="best_overall") as best_run:
        mlflow.log_params(best["hp"])
        mlflow.log_metrics(best["metrics"])

        # Log extra info
        mlflow.log_param("model_name", best["name"])
        mlflow.log_param("epochs", cfg.tuning.epochs)  # or best_hp["epochs"] if tuned

        # Save TensorFlow/Keras model
        mlflow.tensorflow.log_model(
            model=best["model"],
            artifact_path=cfg.logging.artifacts.model  # e.g. "model"
        )

        # Save preprocessing artifacts
        artifact_dir = f"{cfg.logging.artifacts.base_dir}_{best['name']}"
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(artifact_dir, "encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
        
        mlflow.log_dict({"features": features}, 
                f"{cfg.logging.artifacts.preprocessing.base_dir}/features.json")

        mlflow.log_artifacts(artifact_dir, artifact_path=cfg.logging.artifacts.preprocessing.base_dir)

    return best_run.info.run_id



def basic_logging_per_model(model_name, epochs, best_hp, val_metrics):
    with mlflow.start_run(run_name=f"{model_name}_run"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_params(best_hp)
        mlflow.log_metrics(val_metrics)


def basic_comparisson(results):
    with mlflow.start_run(run_name="comparison") as comp_run:
        for res in results:
            prefix = res["name"]
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in res["metrics"].items()})
        # maybe a couple of plots
        comparison_run_id = comp_run.info.run_id  # save if needed