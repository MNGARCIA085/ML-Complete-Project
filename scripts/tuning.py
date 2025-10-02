import hydra
from omegaconf import DictConfig
import mlflow
from src.tuning.tuner import ModelTuner
from src.data.preprocessor import Preprocessor
from src.experiments.logging import log_best_model, basic_logging_per_model, basic_comparisson
from src.experiments.selection import get_best_model
from src.models.factory import get_model_builder
from src.common.plotting import plot_confusion_matrix, plot_roc_curve


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Set MLflow experiment
    mlflow.set_experiment(cfg.tuning.experiment_name)

    print("=== Data Preparation ===")
    preprocessor = Preprocessor(cfg)
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)

    keys = ["train_ds", "val_ds", "scaler", "encoder", "feature_columns", "input_shape"]
    train_ds, val_ds, scaler, encoder, features, input_shape_raw = (prep_results[k] for k in keys)
    input_dim = input_shape_raw[0]

    print("=== Models Tuning ===")
    models_to_tune = ["baseline", "model1"]
    all_results = []

    # Run for each model: log hyperparams + metrics
    for model_name in models_to_tune:
        print(f"=== Training model: {model_name} ===")

        # Model
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model
        build_fn = get_model_builder(model_name)

        # Tuning
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
        basic_logging_per_model(model_name, cfg.tuning.epochs, best_hp.values, val_metrics)

        # prints
        print(model_name)
        print("Best hyperparameters:", best_hp.values)
        print("Validation metrics:", val_metrics)

    # Comparison run: log all metrics, pick best
    basic_comparisson(all_results)

    # get best model
    best = get_best_model(all_results, recall_threshold=cfg.tuning.recall_threshold)

    # ----- Confusion matrix -----
    cm_plot_path = plot_confusion_matrix(best['model'], val_ds, best['name'], "confusion_matrix.png")

    # ----- ROC Curve -----
    roc_curve_plot_path = plot_roc_curve(best['model'], val_ds, best['name'], "roc_curve.png")

    # Final run: store best model + preprocessing + metrics
    best_run_id = log_best_model(
        best=best,
        cfg=cfg,
        scaler=scaler,
        encoder=encoder,
        features=features,
        test_metrics=None,
        cm_plot_path=cm_plot_path,
        roc_curve_plot_path=roc_curve_plot_path,
    )


    # final prints
    print(f"Overall best model: {best['name']}")
    print("Best hyperparameters:", best["hp"])
    print("Best validation metrics:", best["metrics"])
    print("Final best model run_id:", best_run_id)


if __name__ == "__main__":
    main()



