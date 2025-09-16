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
from src.models.model1 import build_compile_model_one # _tuner_2
from src.models.baseline import build_compile_baseline

# Map model names to their builder functions
MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}

# nbote : i canb chaneg to combine data and model

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    print("=== Data Preparation ===")
    # Initialize preprocessor with cfg.data
    preprocessor = Preprocessor(cfg.data)

    # train/val
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)

    # keys
    keys = ["train_ds", "val_ds", "scaler", "encoder", "feature_columns", "input_shape"]
    train_ds, val_ds, scaler, encoder, features, input_shape_raw = (prep_results[k] for k in keys)
    input_dim = input_shape_raw[0]  # keep the first element as before



    print("=== Models Tuning ===")
    results_list = []
    models_to_tune = ["baseline", "model1"]

    for model_name in models_to_tune:
        print(f"=== Training model: {model_name} ===")

        # Compose Hydra config for this model (choosing the appropiate model config file)
        model_cfg = hydra.compose(config_name="config", overrides=[f"model={model_name}"]).model #_tuner

        # Pass the build function for this model
        build_fn = MODEL_BUILDERS[model_name]  # e.g., build_compile_model_one, build_compile_baseline


        print(model_cfg)

        # Create tuner
        
        tuner = ModelTuner(cfg.tuning, model_cfg, build_fn, input_dim)

        # Run tuner
        best_model, best_hp, val_metrics = tuner.run(train_ds, val_ds)

        print(model_name)
        print("Best hyperparameters:", best_hp.values)
        print("Validation metrics:", val_metrics)
        




if __name__ == "__main__":
    main()



"""
global_cfg = hydra.compose(config_name="config/config.yaml")
model_cfg = hydra.compose(config_name="config/model/model1.yaml").model
"""