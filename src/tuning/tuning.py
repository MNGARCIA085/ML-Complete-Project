import hydra
from omegaconf import DictConfig
import mlflow
import json
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.tuning.tuner import ModelTuner
from src.data.preprocessor import Preprocessor
from src.models.model1 import build_compile_model_one_tuner_2
from src.models.baseline import build_compile_baseline_tuner_2

# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline_tuner_2,
    "model1": build_compile_model_one_tuner_2
}

# nbote : i canb chaneg to combine data and model

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Data Preparation ===")
    preprocessor = Preprocessor(cfg.data)
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)
    train_ds = prep_results["train_ds"]
    val_ds = prep_results["val_ds"]
    input_shape = prep_results["input_shape"][0]
    scaler = prep_results["scaler"]
    encoder = prep_results["encoder"]
    features = prep_results["feature_columns"]


    results_list = []

    models_to_train = ["baseline", "model1"]
    input_dim = input_shape  # e.g., train_ds.element_spec[0].shape[-1]

    for model_name in models_to_train:
        print(f"=== Training model: {model_name} ===")

        # Compose Hydra config for this model (choosing the appropiate model config file)
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}_tuner"]).model

        # Pass the build function for this model
        build_fn = MODEL_BUILDERS[model_name]  # e.g., build_compile_model_one_tuner


        print(model_cfg)

        # Create tuner
        
        tuner = ModelTuner(cfg, build_fn, input_dim, model_cfg)

        # Run tuner
        best_model, best_hp, val_metrics = tuner.run(train_ds, val_ds)

        print(model_name)
        print("Best hyperparameters:", best_hp.values)
        print("Validation metrics:", val_metrics)
        




if __name__ == "__main__":
    main()



"""
global_cfg = hydra.compose(config_name="config/config.yaml")

model_cfg = hydra.compose(config_name="config/model/baseline_tuner.yaml").model
"""