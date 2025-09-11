import hydra
from omegaconf import DictConfig
from src.data.preprocessor import Preprocessor
from src.models.baseline import build_compile_baseline
from src.models.model1 import build_compile_model_one


# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    print("=== Data Preparation ===")
    
    # Initialize preprocessor with cfg.data
    preprocessor = Preprocessor(cfg.data)
    train_ds, val_ds = preprocessor.prepare_train_val(cfg.data.data_path)
    test_ds = preprocessor.prepare_test(cfg.data.data_test_path)


    input_shape = preprocessor.get_input_shape()
    print("Input shape for model:", input_shape)  # e.g., (30,)


    # Determine which model to train from cfg.model
    model_name = cfg.model.name

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")


    # Build the model
    model = MODEL_BUILDERS[model_name](cfg.model, input_dim=input_shape[0])
    print(model.summary())




if __name__ == "__main__":
    main()



# python -m scripts.models -m model=baseline,model1 (multirun)



"""
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

models_to_train = ["baseline", "model1"]

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    for model_name in models_to_train:
        # Override the model config for each iteration
        cfg.model = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model
        
        print(f"Training model: {model_name}")
        model = build_model(cfg.model, input_dim=cfg.data.input_dim)
        # Call your training function here
        train(model, cfg.data)

if __name__ == "__main__":
    main()
"""