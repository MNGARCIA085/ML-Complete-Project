import hydra
from omegaconf import DictConfig
import mlflow
from src.training.trainer import ModelTrainer
from src.data.preprocessor import Preprocessor  # assuming your class is in src/data/preprocessor.py

from src.models.model1 import build_compile_model_one
from src.models.baseline import build_compile_baseline




# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}





@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # get data, get model, train model, log results; custom plots (like conf. matrix) and log appropuiately
    # for now, same data


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



    print("Models")
    models_to_train = ["baseline", "model1"]
    for model_name in models_to_train:
        # Override the model config for each iteration
        cfg.model = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model


        # better practice
        #model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model
        #model = MODEL_BUILDERS[model_name](model_cfg, input_dim=input_shape)

        
        # build and compile the appropiate model
        print(f"Training model: {model_name}")
        #model = build_model(cfg.model, input_dim=cfg.data.input_dim)
        model = MODEL_BUILDERS[model_name](cfg.model, input_dim=input_shape)

        print("------Training--------")
        # Init trainer
        trainer = ModelTrainer(cfg)

        # Train
        results = trainer.train(model, train_ds, val_ds)






if __name__ == "__main__":
    main()
