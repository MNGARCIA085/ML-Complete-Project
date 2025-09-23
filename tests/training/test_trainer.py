import pytest
from unittest.mock import MagicMock
from src.common.metrics import compute_f1
from src.training.trainer import ModelTrainer


class DummyTrainingCfg:
    def __init__(self):
        class OptimizerCfg:
            type = "adam"
            class LR:
                default = 0.001
            lr = LR()
        self.optimizer = OptimizerCfg()
        self.epochs = 5
        self.callbacks = ["cb1", "cb2"]


@pytest.fixture
def trainer():
    return ModelTrainer(DummyTrainingCfg())


def test_init_hyperparams_and_callbacks(trainer):
    assert trainer.callbacks == ["cb1", "cb2"]
    assert trainer.hyperparameters == {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 5,
    }


def test_train_calls_fit_and_evaluate(trainer):
    # Mock model
    model = MagicMock()
    mock_history = MagicMock()
    model.fit.return_value = mock_history
    model.evaluate.return_value = {
        "loss": 0.5,
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.6,
    }

    result = trainer.train(model, "train_ds", "val_ds")

    # --- fit was called correctly
    model.fit.assert_called_once_with(
        "train_ds",
        validation_data="val_ds",
        epochs=5,
        callbacks=["cb1", "cb2"],
    )

    # --- evaluate was called correctly
    model.evaluate.assert_called_once_with("val_ds", verbose=0, return_dict=True)

    # --- result contains expected keys
    assert result["model"] == model
    assert result["history"] == mock_history
    assert result["val_loss"] == 0.5
    assert result["val_accuracy"] == 0.8
    assert result["val_precision"] == 0.75
    assert result["val_recall"] == 0.6
    assert result["val_f1"] == compute_f1(0.75, 0.6)
    assert result["hyperparameters"]["epochs"] == 5


def test_evaluate_model_handles_missing_keys(trainer):
    model = MagicMock()
    model.evaluate.return_value = {"loss": 1.0, "accuracy": 0.5}

    result = trainer._evaluate_model(model, "val_ds")

    assert result["val_loss"] == 1.0
    assert result["val_accuracy"] == 0.5
    assert result["val_precision"] == 0.0
    assert result["val_recall"] == 0.0
    assert result["val_f1"] == compute_f1(0.0, 0.0)


"""
This code:
    Uses a DummyTrainingCfg so you don’t depend on Hydra in tests.
    Uses MagicMock instead of real TensorFlow models/datasets (fast).
    Checks that:
        Callbacks & hyperparams are stored correctly.
        fit and evaluate are called with the right args.
        Returned dict contains everything, including val_f1.
        Graceful handling when precision/recall keys are missing.
"""