import hydra
from omegaconf import DictConfig
import mlflow
from src.data.preprocessor import Preprocessor
from src.models.factory import get_model
from src.training.trainer import ModelTrainer
from src.utils.plotting import plot_history_curve,plot_confusion_matrix,plot_comparison
from src.experiments.logging import log_experiment_results,log_model_comparison




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
        model = get_model(model_name, model_cfg, cfg.training, input_dim=input_shape) # model already build and compiled


        # trainer
        trainer = ModelTrainer(cfg.training)

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
            loss_plot_path=loss_plot_path,
            acc_plot_path=acc_plot_path,
            cm_plot_path=cm_plot_path,
            global_cfg=cfg,
        )


        # Store results for comparison
        results_list.append({
            "model_name": model_name,
            "history": results["history"],
            **{m: results[m] for m in metrics}
        })


    # ----- Comparison run -----
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