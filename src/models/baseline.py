from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Input
from src.models.utils import compile_model


# build and compile model
def build_compile_baseline(model_cfg, training_cfg, input_dim, hp=None):
    """
    Builds and compiles a model, optionally using hp for tuning.
    """

    # Learning rate and dropout
    if hp:
        lr = hp.Float("learning_rate", model_cfg.lr.low, model_cfg.lr.high, sampling=model_cfg.lr.sampling)
    else:
        lr = model_cfg.lr.default


    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(model_cfg.hidden_layer_neurons, activation=model_cfg.hidden_layer_activation),
        Dense(1, activation=model_cfg.output_activation)
    ], name=model_cfg.name)

    compile_model(model, training_cfg)
    return model








