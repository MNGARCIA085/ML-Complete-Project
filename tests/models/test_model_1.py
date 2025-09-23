# tests/test_model_one.py
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from src.models.model1 import build_compile_model_one  

class DummyModelCfg:
    name = "model_one"
    layers = [16, 8]  # two hidden layers
    activation_options = ["relu"]
    output_activation = "sigmoid"

    class lr:
        low = 1e-4
        high = 1e-2
        default = 1e-3
        sampling = "log"

    class dropout:
        default = 0.5
        low = 0.1
        high = 0.6
        step = 0.1

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

def test_build_compile_model_one_shapes():
    input_dim = 10
    model = build_compile_model_one(DummyModelCfg, DummyTrainingCfg, input_dim)
    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, 1)
    # 2 dense + dropout + output = 4 layers
    assert any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)

def test_model_one_trains():
    input_dim = 10
    model = build_compile_model_one(DummyModelCfg, DummyTrainingCfg, input_dim)
    x_dummy = np.random.rand(12, input_dim).astype("float32")
    y_dummy = np.random.randint(0,2,size=(12,1))
    history = model.fit(x_dummy, y_dummy, epochs=1, batch_size=4, verbose=0)
    assert "loss" in history.history

def test_hyperparameter_integration():
    class DummyHP:
        def Float(self, name, low, high, **kwargs):
            # return some mid value
            return (low + high) / 2
    input_dim = 10
    model = build_compile_model_one(DummyModelCfg, DummyTrainingCfg, input_dim, hp=DummyHP())
    assert isinstance(model, Sequential)
