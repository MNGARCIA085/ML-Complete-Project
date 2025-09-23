import pandas as pd
from src.experiments.loading import load_best_model
import numpy as np




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
    model, scaler, encoder, features  = load_best_model(cfg.logging, cfg.tuning.experiment_name, 
                    cfg.tuning.recall_threshold, cfg.logging.artifacts.model)
    
    # Convert to DataFrame with proper column names
    X_df = pd.DataFrame([X_raw], columns=features) if X_raw.ndim == 1 else pd.DataFrame(X_raw, columns=features)

    # Apply preprocessing (and other prep. but in this case i will already pass 30 features)
    X_scaled = scaler.transform(X_df)
    #X_prepared = encoder.transform(X_scaled) if encoder is not None else X_scaled
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    return y_pred



"""
naming (good practices)

predict_with_best_model(cfg, X_raw) → if it always uses the best run.

predict_with_run(cfg, run_id, X_raw) → if you want to specify run_id explicitly.
"""