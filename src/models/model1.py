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



def build_compile_model_one_tuner(cfg, input_dim, hp):
    learning_rate = hp.Float("learning_rate", cfg.model.lr.low, cfg.model.lr.high, sampling=cfg.model.lr.sampling)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.model.layers[0], activation=cfg.model.activation_options[0]),
        Dense(cfg.model.layers[1], cfg.model.activation_options[0]),
        Dropout(rate=cfg.model.dropout.default),
        Dense(1, activation=cfg.model.output_activation)
    ], name=cfg.model.name)

    compile_model(model, learning_rate)
    return model


def build_compile_model_one_tuner_2(cfg, input_dim, hp):
    learning_rate = hp.Float("learning_rate", cfg.lr.low, cfg.lr.high, sampling=cfg.lr.sampling)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.layers[0], activation=cfg.activation_options[0]),
        Dense(cfg.layers[1], cfg.activation_options[0]),
        Dropout(rate=cfg.dropout.default),
        Dense(1, activation=cfg.output_activation)
    ], name=cfg.name)

    compile_model(model, learning_rate)
    return model




"""
Write model just once
def build_compile_model(cfg, input_dim, hp=None):
    Builds and compiles a model, optionally using hp for tuning.
    
    # Learning rate: use hp if provided, else default or midpoint
    if hp:
        lr = hp.Float("learning_rate", cfg.model.lr.low, cfg.model.lr.high, sampling=cfg.model.lr.sampling)
    else:
        lr = (cfg.model.lr.low + cfg.model.lr.high) / 2  # or cfg.model.lr.default

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.model.layers[0], activation=cfg.model.activation_options[0]),
        Dense(cfg.model.layers[1], activation=cfg.model.activation_options[0]),
        Dropout(rate=cfg.model.dropout.default),
        Dense(1, activation=cfg.model.output_activation)
    ], name=cfg.name)

    compile_model(model, lr)
    return model
"""


"""
def build_compile_model(cfg, input_dim, hp=None):
    Builds and compiles a model, optionally using hp for tuning.

    # Learning rate
    if hp:
        lr = hp.Float("learning_rate", cfg.model.lr.low, cfg.model.lr.high, sampling=cfg.model.lr.sampling)
    else:
        lr = (cfg.model.lr.low + cfg.model.lr.high) / 2  # or cfg.model.lr.default

    # Dropout rate
    if hp and hasattr(cfg.model.dropout, "tunable") and cfg.model.dropout.tunable:
        rate = hp.Float(
            "dropout_rate",
            cfg.model.dropout.low,
            cfg.model.dropout.high,
            step=cfg.model.dropout.step
        )
    else:
        rate = cfg.model.dropout.default

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.model.layers[0], activation=cfg.model.activation_options[0]),
        Dense(cfg.model.layers[1], activation=cfg.model.activation_options[0]),
        Dropout(rate=rate),
        Dense(1, activation=cfg.model.output_activation)
    ], name=cfg.name)

    compile_model(model, lr)
    return model

model:
  name: my_model
  layers:
    - 64
    - 32
  activation_options:
    - relu
  output_activation: sigmoid
  lr:
    low: 0.001
    high: 0.01
    sampling: linear
  dropout:
    default: 0.2
    tunable: true
    low: 0.0
    high: 0.5
    step: 0.1

"""