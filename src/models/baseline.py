from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Input
from src.models.utils import compile_model


# build model
def build_baseline(cfg, input_dim=None):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(cfg.hidden_layer_neurons, cfg.hidden_layer_activation),
        Dense(1, activation=cfg.output_activation)
    ],name=cfg.name)
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
