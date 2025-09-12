import hydra
from omegaconf import DictConfig
import mlflow

from src.training.trainer import ModelTrainer

from src.data.preprocessor import Preprocessor  # assuming your class is in src/data/preprocessor.py




from src.models.baseline import build_compile_baseline



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):





    print("=== Data Preparation ===")
    # Initialize preprocessor with cfg.data
    preprocessor = Preprocessor(cfg.data)
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)
    train_ds = prep_results["train_ds"]
    val_ds = prep_results["val_ds"]
    input_shape = prep_results["input_shape"][0]
    scaler = prep_results["scaler"]
    encoder = prep_results["encoder"]
    features = prep_results["feature_columns"]
    #model_name = cfg.model.name; note it uses default file; help with multirun

    # build and compile model    
    model = build_compile_baseline(cfg.model, input_dim=input_shape)
    print(model.summary())


    #model_fn = build_compile_baseline()


    # Run training
    print("=== Training ===")

    # Init trainer
    trainer = ModelTrainer(cfg)

    # Train
    results = trainer.train(model, train_ds, val_ds)



    print('nico')


    # Log results to MLflow
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=results["model_name"]):
        mlflow.log_params(results["hyperparameters"])
        mlflow.log_param("model_name", results["model_name"])
        mlflow.log_metrics({
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
            "val_precision": results["val_precision"],
            "val_recall": results["val_recall"],
            "val_f1": results["val_f1"],
        })
        mlflow.tensorflow.log_model(results["model"], artifact_path="model")


if __name__ == "__main__":
    main()
