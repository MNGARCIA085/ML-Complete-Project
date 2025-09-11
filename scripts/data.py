import hydra
from omegaconf import DictConfig
from src.data.preprocessor import Preprocessor  # assuming your class is in src/data/preprocessor.py


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("=== Data Preparation ===")

    #print(cfg.data)

    # Initialize preprocessor
    preprocessor = Preprocessor(cfg.data)

    # Train + val
    train_ds, val_ds = preprocessor.prepare_train_val(cfg.data.data_path)
    print(f"Train dataset batches: {len(list(train_ds))}")
    print(f"Val dataset batches: {len(list(val_ds))}")

    # Test
    test_ds = preprocessor.prepare_test(cfg.data.data_test_path)
    print(f"Test dataset batches: {len(list(test_ds))}")


    for t in train_ds.take(1):
        print(t[0].shape, t[1].shape)


if __name__ == "__main__":
    main()


#python -m scripts.data
#python -m scripts.data data.batch_size=64

"""
python src/data.py data.batch_size=64 data.preprocessing.val_size=0.2

python src/data.py data.data_path="data/raw/new_dataset.csv"
print(OmegaConf.to_yaml(cfg))
"""