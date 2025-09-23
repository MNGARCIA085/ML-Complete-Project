# test_model.py
import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from src.models.baseline import build_compile_baseline, compile_model

class DummyModelCfg:
    name = "baseline"
    hidden_layer_neurons = 8
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
    metrics = ["accuracy", "precision", "recall"]


def test_build_compile_baseline_shapes():
    input_dim = 5
    model = build_compile_baseline(DummyModelCfg, DummyTrainingCfg, input_dim)
    
    # Check input shape
    assert model.input_shape == (None, input_dim)
    # Check output shape
    assert model.output_shape == (None, 1)
    # Check number of layers
    assert len(model.layers) == 2  # Dense + Dense

def test_model_compiles():
    input_dim = 5
    model = build_compile_baseline(DummyModelCfg, DummyTrainingCfg, input_dim)
    # Try a forward pass
    import numpy as np
    x_dummy = np.random.rand(3, input_dim).astype("float32")
    y_dummy = np.random.randint(0,2,size=(3,1))
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)

def test_hyperparameter_integration():
    class DummyHP:
        def Float(self, name, low, high, sampling):
            # just return a mid-value for testing
            return 0.001
    input_dim = 5
    model = build_compile_baseline(DummyModelCfg, DummyTrainingCfg, input_dim, hp=DummyHP())
    assert isinstance(model, Sequential)
