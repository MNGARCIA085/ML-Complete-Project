import pandas as pd
import numpy as np
from src.experiments.loading import load_best_model
from src.data.preprocessor import Preprocessor




def predict_with_best_model(cfg, X_raw: np.ndarray):
    """
    Given raw input data, load the appropriate model & preprocessing from MLflow
    and return predictions.
    
    Args:
        run_id (str): MLflow run ID
        X_raw (np.ndarray): raw input data, shape (n_features,) or (n_samples, n_features)
    
    Returns:
        y_pred: predictions
    """

    # Load model, scaler.....................
    model, scaler, encoder, features  = load_best_model(cfg.logging, cfg.tuning.experiment_name, 
                    cfg.tuning.recall_threshold, cfg.logging.artifacts.model)
    
    
    # Preprocessing data
    preprocessor = Preprocessor(cfg)
    X_prep = preprocessor.prepare_sample(X_raw, scaler, encoder, features)

    
    # Predict
    y_pred = model.predict(X_prep)
    
    return y_pred




"""
naming (good practices)

predict_with_best_model(cfg, X_raw) → if it always uses the best run.

predict_with_run(cfg, run_id, X_raw) → if you want to specify run_id explicitly.
"""