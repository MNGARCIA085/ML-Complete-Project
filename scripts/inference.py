import hydra
from omegaconf import DictConfig
import numpy as np
from src.inference.inference import predict_with_best_model


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    # example raw data
    X_new_sample = np.random.rand(30)  # replace with actual patient data

    # prediction
    y_pred = predict_with_best_model(cfg, X_new_sample)
    print(y_pred)


# main
if __name__==main():
    main()