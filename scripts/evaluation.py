from src.experiments.loading import load_best_model
import hydra
from omegaconf import DictConfig
from src.data.preprocessor import Preprocessor



# main
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get best model, scaler, encoder.......
    model, scaler, encoder, features = load_best_model(cfg.logging, cfg.tuning.experiment_name, cfg.tuning.recall_threshold, cfg.logging.artifacts.model)

    print(features)
    #features = data["features"]

    # prepare test
    preprocessor = Preprocessor(cfg)
    test_ds = preprocessor.prepare_test(cfg.data.data_test_path, scaler, encoder)
    
    # evaluate
    model.evaluate(test_ds)


# main
if __name__==main():
    main()