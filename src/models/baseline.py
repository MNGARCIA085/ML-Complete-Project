from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Input
from src.models.utils import compile_model


# build model
def build_baseline(cfg, input_dim=None):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units=cfg.hidden_layer_neurons, activation=cfg.hidden_layer_activation),
        Dense(units=1, activation=cfg.output_activation)
    ], name=cfg.name)
    # put compile here; i dont need it separate
    return model


# not using a for for architectures that may be noy the same


# build and compile
def build_compile_baseline(cfg, input_dim=None):
    model = build_baseline(cfg, input_dim)
    compile_model(model)
    return model


# for keras tuner
def build_compile_tuner(hp):
    dropout_rate = hp.Float("dropout_rate", 0.0, 0.5, step=0.1)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model = build_baseline(dropout_rate=dropout_rate)

    # Recompile model with tunable learning rate
    compile_model(model, learning_rate)
    return model




def build_compile_baseline_tuner(cfg, input_dim, hp):
    learning_rate = hp.Float("learning_rate", cfg.model.lr.low, cfg.model.lr.high, sampling=cfg.model.lr.sampling)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units=cfg.model.hidden_layer_neurons, activation=cfg.model.hidden_layer_activation),
        Dense(units=1, activation=cfg.model.output_activation)
    ], name=cfg.model.name)

    compile_model(model, learning_rate)
    return model



def build_compile_baseline_tuner_2(cfg, input_dim, hp):
    learning_rate = hp.Float("learning_rate", cfg.lr.low, cfg.lr.high, sampling=cfg.lr.sampling)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units=cfg.hidden_layer_neurons, activation=cfg.hidden_layer_activation),
        Dense(units=1, activation=cfg.output_activation)
    ], name=cfg.name)

    compile_model(model, learning_rate)
    return model
