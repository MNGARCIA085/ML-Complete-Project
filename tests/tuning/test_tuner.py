import pytest
from unittest.mock import MagicMock, patch
from src.tuning.tuner import ModelTuner
from src.common.metrics import compute_f1


class DummyCfg(dict):
    """Allows attribute and dict access for configs."""
    def __getattr__(self, item):
        return self[item] if item in self else None


@pytest.fixture
def tuner_instance():
    tuning_cfg = DummyCfg(max_trials=1, executions_per_trial=1, epochs=2, patience=3)
    training_cfg = DummyCfg()
    model_cfg = DummyCfg()

    def dummy_build_model_fn(model_cfg, training_cfg, input_dim, hp=None):
        model = MagicMock()
        model.fit.return_value = None
        model.evaluate.return_value = [0.5, 0.8, 0.75, 0.6]  # loss, acc, precision, recall
        model.optimizer = MagicMock()
        model.optimizer.__class__.__name__ = "Adam"
        model.optimizer.learning_rate = 0.001
        return model

    return ModelTuner(
        tuning_cfg, training_cfg, model_cfg,
        dummy_build_model_fn, input_dim=10, seed=123
    )


def test_create_tuner_returns_randomsearch(tuner_instance):
    with patch("src.tuning.tuner.kt.RandomSearch") as mock_random_search:
        tuner_instance._create_tuner()
        assert mock_random_search.called
        _, kwargs = mock_random_search.call_args
        assert kwargs["max_trials"] == 1
        assert kwargs["executions_per_trial"] == 1
        assert kwargs["objective"] == "val_accuracy"


def test_train_best_model_calls_fit(tuner_instance):
    model = MagicMock()
    model.fit.return_value = None

    history, trained_epochs = tuner_instance._train_best_model(
        model, "train_ds", "val_ds", epochs=2
    )

    model.fit.assert_called_once()
    assert isinstance(history, dict)  # comes from HistoryCapture
    assert isinstance(trained_epochs, int)


def test_evaluate_model_computes_metrics(tuner_instance):
    model = MagicMock()
    model.evaluate.return_value = [0.5, 0.8, 0.75, 0.6]

    metrics = tuner_instance._evaluate_model(model, "val_ds")

    assert metrics["loss"] == 0.5
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.6
    assert metrics["f1_score"] == compute_f1(0.75, 0.6)


def test_collect_extra_hyperparams_reads_optimizer_and_lr(tuner_instance):
    model = MagicMock()
    model.optimizer = MagicMock()
    model.optimizer.__class__.__name__ = "Adam"
    model.optimizer.learning_rate = 0.001

    # Fake train_ds with batch_size
    train_ds = MagicMock()
    batch_size_tensor = MagicMock()
    batch_size_tensor.numpy.return_value = 32
    train_ds._batch_size = batch_size_tensor

    extra = tuner_instance._collect_extra_hyperparams(model, train_ds, trained_epochs=5)

    assert extra["trained_epochs"] == 5
    assert extra["batch_size"] == 32
    assert extra["optimizer"] == "Adam"
    assert extra["final_learning_rate"] == 0.001


def test_run_executes_full_pipeline(tuner_instance):
    # Patch RandomSearch so it doesn't actually run search
    with patch("src.tuning.tuner.kt.RandomSearch") as mock_random_search:
        mock_tuner = MagicMock()
        mock_tuner.get_best_models.return_value = [MagicMock()]
        mock_tuner.get_best_hyperparameters.return_value = [MagicMock()]
        mock_random_search.return_value = mock_tuner

        # Ensure model returned by get_best_models has evaluate
        best_model = mock_tuner.get_best_models.return_value[0]
        best_model.evaluate.return_value = [0.5, 0.8, 0.75, 0.6]
        best_model.optimizer = MagicMock()
        best_model.optimizer.__class__.__name__ = "Adam"
        best_model.optimizer.learning_rate = 0.001

        result = tuner_instance.run("train_ds", "val_ds")

        assert isinstance(result, tuple)
        best_model, best_hp, metrics = result
        assert "accuracy" in metrics
        assert "f1_score" in metrics
