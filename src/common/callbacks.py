import tensorflow as tf

# Capture history
class HistoryCapture(tf.keras.callbacks.Callback):
    """Capture training history in a dict format."""
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))