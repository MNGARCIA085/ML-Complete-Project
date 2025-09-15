import hydra
from omegaconf import DictConfig
from src.data.preprocessor import Preprocessor
from src.models.baseline import build_compile_baseline_tuner
from src.models.model1 import build_compile_model_one_tuner


# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline_tuner
    "model1": build_compile_model_one_tuner
}



class DummyHP:
    def Float(self, name, low, high, sampling=None):
        # Just return a fixed value (midpoint)
        return (low + high) / 2

#The learning_rate will be set to (cfg.model.lr.low + cfg.model.lr.high) / 2.








@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    print("=== Data Preparation ===")
    
    # Initialize preprocessor with cfg.data
    preprocessor = Preprocessor(cfg.data)
    
    # train/val
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)
    train_ds = prep_results["train_ds"]
    val_ds = prep_results["val_ds"]
    input_shape = prep_results["input_shape"][0] # e.g., (30,), it should be use with input_shape[0]
    scaler = prep_results["scaler"]
    encoder = prep_results["encoder"]
    features = prep_results["feature_columns"]

    # test
    test_ds = preprocessor.prepare_test(cfg.data.data_test_path)

    #input_shape = preprocessor.get_input_shape()
    print("Input shape for model:", input_shape)  

    # Determine which model to train from cfg.model
    model_name = cfg.model.name

    print('dsdsffddfsfddfs', cfg.model)


    # Build the model
    hp = DummyHP()
    # model = build_compile_model_one_tuner(cfg, input_dim=input_shape, hp=hp)
    model = MODEL_BUILDERS[cfg.model.name](cfg, input_dim=input_shape, hp=hp)
    print(model.summary())





if __name__ == "__main__":
    main()



#python -m src.tuning.testing_models -m model=model1_tuner  ; uses thta specific config file
#python -m src.tuning.testing_models -m model=baseline_tuner,model_1_tuner
# without anything it gives an error because by defaut it uses baseline



"""
better coding:

# Initialize preprocessor with cfg.data
preprocessor = Preprocessor(cfg.data)

# train/val
prep_results = preprocessor.prepare_train_val(cfg.data.data_path)

# Unpack all relevant items at once
train_ds, val_ds = prep_results["train_ds"], prep_results["val_ds"]
input_shape, scaler, encoder, features = (
    prep_results["input_shape"][0],
    prep_results["scaler"],
    prep_results["encoder"],
    prep_results["feature_columns"],
)

keys = ["train_ds", "val_ds", "scaler", "encoder", "feature_columns", "input_shape"]
train_ds, val_ds, scaler, encoder, features, input_shape_raw = (prep_results[k] for k in keys)
input_shape = input_shape_raw[0]  # keep the first element as before
"""








"""
fixed value
def build_compile_model_one_tuner(cfg, input_dim, learning_rate_value=None):
    if learning_rate_value is None:
        learning_rate_value = (cfg.model.lr.low + cfg.model.lr.high) / 2

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.model.layers[0], activation=cfg.model.activation_options[0]),
        Dense(cfg.model.layers[1], cfg.model.activation_options[0]),
        Dropout(rate=cfg.model.dropout.default),
        Dense(1, activation=cfg.model.output_activation)
    ], name=cfg.name)

    compile_model(model, learning_rate_value)
    return model
"""