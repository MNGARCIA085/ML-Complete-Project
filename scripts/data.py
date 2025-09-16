import hydra
from omegaconf import DictConfig
from src.data.preprocessor import Preprocessor


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("=== Data Preparation ===")

    #print(cfg.data)

    # Initialize preprocessor with cfg.data
    preprocessor = Preprocessor(cfg.data)

    # train/val
    prep_results = preprocessor.prepare_train_val(cfg.data.data_path)

    # Unpack all relevant items at once
    train_ds, val_ds = prep_results["train_ds"], prep_results["val_ds"]
    input_shape, scaler, encoder, features = (
        prep_results["input_shape"][0],
        prep_results["scaler"],
        prep_results["encoder"],
        prep_results["feature_columns"],
    )


    # scaler
    print(scaler)
    print("Means:", scaler.mean_)          # per-feature mean
    print("Scale (std):", scaler.scale_)   # per-feature standard deviation
    print("Var:", scaler.var_)             # per-feature variance
    print("n_samples_seen:", scaler.n_samples_seen_)

    # basic prints
    print(f"Train dataset batches: {len(list(train_ds))}")
    print(f"Val dataset batches: {len(list(val_ds))}")

    # Test
    test_ds = preprocessor.prepare_test(cfg.data.data_test_path)
    print(f"Test dataset batches: {len(list(test_ds))}")


    for t in train_ds.take(1):
        print(t[0].shape, t[1].shape)


if __name__ == "__main__":
    main()


"""
python -m scripts.data
python -m scripts.data data.batch_size=64
"""