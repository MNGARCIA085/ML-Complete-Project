# tests/test_factory.py
import pytest
import tensorflow as tf
from src.models.factory import get_model, get_model_builder

# Dummy configs reused
class DummyModelCfg:
    name = "baseline"
    hidden_layer_neurons = 4
    hidden_layer_activation = "relu"
    output_activation = "sigmoid"
    class lr:
        low = 1e-4
        high = 1e-2
        default = 1e-3
        sampling = "log"

class DummyTrainingCfg:
    class optimizer:
        type = "adam"
        class lr:
            default = 0.001
            low = 1e-4
            high = 1e-2
            sampling = "log"

    loss = "binary_crossentropy"
    metrics = ["accuracy"]

def test_get_model_returns_compiled_model():
    input_dim = 5
    model = get_model("baseline", DummyModelCfg, DummyTrainingCfg, input_dim)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, input_dim)

def test_get_model_invalid_name():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("invalid_model", DummyModelCfg, DummyTrainingCfg, 5)

def test_get_model_builder_returns_fn():
    builder = get_model_builder("baseline")
    assert callable(builder)
    # builder should be the actual build function
    model = builder(DummyModelCfg, DummyTrainingCfg, 5)
    assert isinstance(model, tf.keras.Model)

def test_get_model_builder_invalid_name():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_builder("not_a_model")
