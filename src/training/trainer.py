from src.utils.metrics import compute_f1


class ModelTrainer:
    """
    Trainer class for Keras/TensorFlow models.

    This class handles training and evaluation of an already built and compiled model.
    It stores training callbacks and hyperparameters from a training configuration.

    Attributes:
        callbacks (list): List of Keras callbacks to use during training.
        hyperparameters (dict): Dictionary storing optimizer type, learning rate, and epochs.
    """

    def __init__(self, training_cfg):
        # Use callbacks from the tuning configuration, default to empty list
        self.callbacks =  training_cfg.callbacks or []

        # Store hyperparameters from the tuning configuration
        self.hyperparameters = {
            "optimizer": training_cfg.optimizer.type,
            "learning_rate": training_cfg.optimizer.lr.default,
            "epochs": training_cfg.epochs,
        }


    def train(self, model, train_ds, val_ds):  # already build and compile model; for tuning I can't do this
        """
        Train and evaluate a model.

        Args:
            model: Pre-built and compiled Keras model.
            train_ds: Training dataset (tf.data.Dataset or similar).
            val_ds: Validation dataset (tf.data.Dataset or similar).

        Returns:
            dict: Contains the trained model, training history, evaluation metrics, and hyperparameters.
        """

        # Train the model and obtain the history object
        history = self._train(model, train_ds, val_ds) 

        # Evaluate the model on the validation dataset
        val_metrics = self._evaluate_model(model, val_ds)

        # Return a dictionary with model, history, metrics, and hyperparameters
        return {
            "model": model,
            "history": history,
            **val_metrics,
            "hyperparameters": self.hyperparameters,
        }

    def _train(self, model, train_ds, val_ds):
        """
        Internal method to fit the model.

        Args:
            model: Pre-built and compiled Keras model.
            train_ds: Training dataset.
            val_ds: Validation dataset.

        Returns:
            History: Keras History object from model.fit().
        """
        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.hyperparameters["epochs"],
            callbacks=self.callbacks,
        )

    def _evaluate_model(self, model, val_ds):
        """
        Evaluate the model on the validation dataset.

        Returns
        -------
        dict
            Dictionary containing loss, accuracy, precision, recall, and F1 score.
        """
        val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)

        # Handle possible key name variations
        precision = val_metrics.get("precision") or val_metrics.get("precision_1") or 0.0
        recall = val_metrics.get("recall") or val_metrics.get("recall_1") or 0.0

        return {
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": compute_f1(precision, recall),
        }







"""
✅ Advantages:

Trainer is fully agnostic to model building/compilation.

Works with any pre-built model (baseline, MLP, CNN…).

Hydra config still controls epochs, callbacks, and hyperparameters logging.
"""