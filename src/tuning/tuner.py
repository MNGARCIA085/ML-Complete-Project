import yaml
import time
import tensorflow as tf
import kerastuner as kt
import numpy as np
from src.common.metrics import compute_f1
from src.common.callbacks import HistoryCapture
from src.common.utils import set_seed



class ModelTuner:
    # constructor
    def __init__(self, tuning_cfg, training_cfg, model_cfg, build_model_fn, input_dim, seed=42):

        # configs
        self.tuning_cfg = tuning_cfg
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg

        # seed
        self.seed = seed

        # model
        self.build_model_fn = build_model_fn
        self.input_dim = input_dim

        # tuning
        self.max_trials = tuning_cfg.get("max_trials", 2)
        self.executions_per_trial = tuning_cfg.get("executions_per_trial", 1)
        self.epochs = tuning_cfg.get("epochs", 2)
        self.patience = tuning_cfg.get("patience", 5)
        

    # run
    def run(self, train_ds, val_ds):
        set_seed(self.seed)
        start_time = time.time()

        tuner = self._create_tuner()

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )

        # search
        tuner.search(train_ds, validation_data=val_ds,
                     epochs=self.epochs, callbacks=[early_stopping_cb])

        # Best model + hyperparameters
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Retrain
        history, trained_epochs = self._train_best_model(
            best_model, train_ds, val_ds, self.epochs, early_stopping_cb,
        )

        # Evaluate
        val_metrics_dict = self._evaluate_model(best_model, val_ds)

        # Collect extra hyperparameters
        extra_hyperparams = self._collect_extra_hyperparams(best_model, train_ds, trained_epochs)

        elapsed_time = time.time() - start_time

        # return
        return best_model, best_hp, val_metrics_dict

    # ---------------- HELPER METHODS ----------------

    def _create_tuner(self):
        return kt.RandomSearch(
            lambda hp: self.build_model_fn(self.model_cfg, self.training_cfg, self.input_dim, hp),
            objective="val_accuracy",
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory="tuner_results",
            project_name=self.build_model_fn.__name__,
            overwrite=True
        )

    def _train_best_model(self, model, train_ds, val_ds, epochs, *callbacks):
        history_cb = HistoryCapture()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[*callbacks, history_cb],
            verbose=0
        )
        trained_epochs = len(history_cb.history.get("loss", []))
        return history_cb.history, trained_epochs

    def _evaluate_model(self, model, val_ds):
        val_metrics = model.evaluate(val_ds, verbose=0)
        return {
            "loss": val_metrics[0],
            "accuracy": val_metrics[1],
            "precision": val_metrics[2],
            "recall": val_metrics[3],
            "f1_score": compute_f1(val_metrics[2], val_metrics[3])
        }

    def _collect_extra_hyperparams(self, model, train_ds, trained_epochs):
        batch_size_tensor = getattr(train_ds, "_batch_size", None)
        batch_size = int(batch_size_tensor.numpy()) if batch_size_tensor is not None else "unknown"
        optimizer = type(model.optimizer).__name__
        learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))

        return {
            "trained_epochs": trained_epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "final_learning_rate": learning_rate,
        }


"""
global_cfg = hydra.compose(config_name="config/config.yaml")
model_cfg = hydra.compose(config_name="config/model/model1.yaml").model
"""