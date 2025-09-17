import tensorflow as tf

def compile_model(model, training_cfg):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_cfg.optimizer.lr.default),
        loss=training_cfg.loss,
        metrics=list(training_cfg.metrics)
    )