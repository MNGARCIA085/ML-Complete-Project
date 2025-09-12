

# compute F1-score
def compute_f1(precision, recall):
    """Compute F1"""
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


class ModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.epochs = cfg.training.epochs
        self.hyperparameters = dict(cfg.training.hyperparameters)
        self.callbacks = cfg.training.callbacks or []

    def train(self, model, train_ds, val_ds):

        # Fill missing hyperparameters
        self._fill_missing_hyperparameters()

        # Train
        history = self._train(model, train_ds, val_ds)

        # Evaluate
        val_metrics = self._evaluate_model(model, val_ds)

        return {
            "model": model,
            "history": history,
            **val_metrics,
            "hyperparameters": self.hyperparameters,
            "model_name": self.model_name,
        }

    def _train(self, model, train_ds, val_ds):
        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=self.callbacks,
        )

    def _evaluate_model(self, model, val_ds):
        val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
        precision = val_metrics.get("precision") or val_metrics.get("precision_1") or 0.0
        recall = val_metrics.get("recall") or val_metrics.get("recall_1") or 0.0
        return {
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": compute_f1(precision, recall),
        }

    def _fill_missing_hyperparameters(self):
        if "optimizer" not in self.hyperparameters or "learning_rate" not in self.hyperparameters:
            opt_info = extract_optimizer_info(self.model)
            self.hyperparameters.setdefault("optimizer", opt_info["optimizer"])
            self.hyperparameters.setdefault("learning_rate", opt_info["learning_rate"])
        self.hyperparameters["epochs"] = self.epochs



"""
✅ Advantages:

Trainer is fully agnostic to model building/compilation.

Works with any pre-built model (baseline, MLP, CNN…).

Hydra config still controls epochs, callbacks, and hyperparameters logging.
"""