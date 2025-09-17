from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.models.utils import compile_model


def build_compile_model_one(model_cfg, training_cfg, input_dim, hp=None):
    """
    Builds and compiles a model, optionally using hp for tuning.
    """

    # Learning rate and dropout
    if hp:
        lr = hp.Float("learning_rate", model_cfg.lr.low, model_cfg.lr.high, sampling=model_cfg.lr.sampling)
        rate = hp.Float(
            "dropout_rate",
            model_cfg.dropout.low,
            model_cfg.dropout.high,
            step=model_cfg.dropout.step
        )
    else:
        lr = model_cfg.lr.default
        rate = model_cfg.dropout.default


    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(model_cfg.layers[0], activation=model_cfg.activation_options[0]),
        Dense(model_cfg.layers[1], activation=model_cfg.activation_options[0]),
        Dropout(rate=rate),
        Dense(1, activation=model_cfg.output_activation)
    ], name=model_cfg.name)

    compile_model(model, training_cfg)
    return model

