# test_preprocessor.py
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from src.data.preprocessor import Preprocessor


"""
These tests cover:

split_data and scaling (fit_scaler + transform_with_scaler).
tf_dataset creation and batch shapes.
prepare_sample for inference.
prepare_test for evaluation.

"""



class DummyCfg:
    seed = 42
    batch_size = 4
    class data:
        target_col = "target"
        class preprocessing:
            val_size = 0.2
            encode_categorical = {"M": 1, "B": 0}

@pytest.fixture
def dummy_csv(tmp_path):
    df = pd.DataFrame({
        "id": range(10),
        "feature1": [1,2,3,4,5,6,7,8,9,10],
        "feature2": [10,9,8,7,6,5,4,3,2,1],
        "target": ["M","B","M","B","M","B","M","B","M","B"]  # 5 per class
    })
    path = tmp_path / "dummy.csv"
    df.to_csv(path, index=False)
    return path

def test_split_and_scaling(dummy_csv):
    pre = Preprocessor(DummyCfg)
    df = pre.load_data(dummy_csv)
    X_train, X_val, y_train, y_val = pre.split_data(df)

    # check shapes
    assert X_train.shape[0] + X_val.shape[0] == df.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == df.shape[0]

    # check scaler
    X_train_scaled = pre.fit_scaler(X_train)
    np.testing.assert_almost_equal(X_train_scaled.mean().values, [0,0], decimal=6)

    X_val_scaled = pre.transform_with_scaler(X_val)
    assert X_val_scaled.shape == X_val.shape

def test_tf_dataset(dummy_csv):
    pre = Preprocessor(DummyCfg)
    df = pre.load_data(dummy_csv)
    X_train, X_val, y_train, y_val = pre.split_data(df)
    pre.fit_scaler(X_train)
    ds = pre.tf_dataset(pre.transform_with_scaler(X_train), y_train)
    
    # Check dataset type and batch
    assert isinstance(ds, tf.data.Dataset)
    for batch_x, batch_y in ds.take(1):
        assert batch_x.shape[1] == X_train.shape[1]

def test_prepare_sample(dummy_csv):
    pre = Preprocessor(DummyCfg)
    df = pre.load_data(dummy_csv)
    X_train, _, _, _ = pre.split_data(df)
    pre.fit_scaler(X_train)
    
    sample = X_train.iloc[0].values
    scaled = pre.prepare_sample(sample, pre.scaler, pre.encoder, pre.feature_columns)
    assert scaled.shape[1] == X_train.shape[1]

def test_prepare_test(dummy_csv):
    pre = Preprocessor(DummyCfg)
    df = pre.load_data(dummy_csv)
    X_train, _, y_train, _ = pre.split_data(df)
    pre.fit_scaler(X_train)

    test_ds = pre.prepare_test(dummy_csv, pre.scaler, pre.encoder)
    assert isinstance(test_ds, tf.data.Dataset)




#pytest tests/preprocessing
#pytest -q tests/preprocessing/test_preprocessor.py -v