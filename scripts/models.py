import hydra
from omegaconf import DictConfig
from src.data.preprocessor import Preprocessor
from src.models.factory import get_model




@hydra.main(config_path="../config", config_name="config", version_base="1.3")
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

    #input_shape = preprocessor.get_input_shape()
    print("Input shape for model:", input_shape)  

    # Determine which model to train from cfg.model
    model_name = cfg.model.name

    # Build and compile the model    
    model = get_model(model_name, cfg.model, cfg.training, input_dim=input_shape)
    print(model.summary())

    # More info about the model

    # Print optimizer
    print("----------Optimizer--------------")
    print(model.optimizer)        # shows optimizer object
    print(model.optimizer.get_config())  # shows optimizer config (lr, beta1, etc.)

    # for the metrics i might need to run the model for at least one epoch
    """
    # Metrics — version-safe way
    # Use evaluate on a *tiny batch* to see what metrics would be computed
    dummy_x, dummy_y = next(iter(train_ds.take(1)))  # take 1 batch
    val_metrics = model.evaluate(dummy_x, dummy_y, verbose=0, return_dict=True)
    print(val_metrics)
    """

if __name__ == "__main__":
    main()



"""
python -m scripts.models  # baseline by default
python -m scripts.models -m model=model1
python -m scripts.models -m model=baseline,model1
"""


# python -m scripts.models -m model=baseline,model1 (multirun)

