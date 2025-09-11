from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.models.utils import compile_model


# build model
def build_model(cfg, input_dim=None):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.layers[0], activation=cfg.activation_options[0]),
        Dense(cfg.layers[1], cfg.activation_options[0]),
        Dropout(rate=cfg.dropout.default),
        Dense(1, activation=cfg.output_activation)
    ], name=cfg.name)
    return model


# build and compile
def build_compile_model_one(cfg, input_dim=None):
    model = build_model(cfg, input_dim)
    compile_model(model)
    return model





# for keras tuner (later!!!)
def build_compile_model_one_tuner(hp):
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model = build_model()
    compile_model(model, learning_rate)
    return model