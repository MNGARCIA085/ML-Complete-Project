import os
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, cfg):
        self.random_state = cfg.seed
        self.val_size = cfg.preprocessing.val_size
        self.target_col = cfg.target_col
        self.batch_size = cfg.batch_size
        self.scaler = None
        self.encoder = cfg.preprocessing.encode_categorical or {"M": 1, "B": 0}
        self.feature_columns = None
        self.input_shape = None  # NEW
        self.save_dir = cfg.preprocessing.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self, filepath):
        return pd.read_csv(filepath)

    def split_data(self, df):
        # Drop unwanted cols
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()

        # Encode labels
        labels = df[self.target_col].map(self.encoder)

        # Features
        features = df.drop(columns=[self.target_col])
        self.feature_columns = features.columns.tolist()

        # input shape
        self.input_shape = (features.shape[1],)  # store input shape

        # Split before scaling
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=self.val_size,
            stratify=labels,
            random_state=self.random_state
        )

        return X_train, X_val, y_train, y_val

    def fit_scaler(self, X_train):
        self.scaler = StandardScaler() # cambair dsp. x el de la config!!!!
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        # Save scaler for later
        joblib.dump(self.scaler, os.path.join(self.save_dir, "scaler.pkl"))
        return X_train_scaled

    def transform_with_scaler(self, X):
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

    def tf_dataset(self, X, y, shuffle=True, batch_size=None):
        bs = batch_size or self.batch_size
        ds = tf.data.Dataset.from_tensor_slices(
            (X.values.astype("float32"), y.values.astype("int32"))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        return ds.batch(bs)

    def prepare_train_val(self, filepath):
        df = self.load_data(filepath)
        X_train, X_val, y_train, y_val = self.split_data(df) # preprocess and split data
        X_train_scaled = self.fit_scaler(X_train)
        X_val_scaled = self.transform_with_scaler(X_val)

        train_ds = self.tf_dataset(X_train_scaled, y_train, shuffle=True)
        val_ds = self.tf_dataset(X_val_scaled, y_val, shuffle=False)

        return train_ds, val_ds

    def prepare_test(self, filepath):
        # Load scaler if not in memory
        if self.scaler is None:
            self.scaler = joblib.load(os.path.join(self.save_dir, "scaler.pkl"))

        df = self.load_data(filepath)
        # Process features/labels but do NOT fit scaler
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()
        labels = df[self.target_col].map(self.encoder)
        features = df.drop(columns=[self.target_col])
        features_scaled = self.transform_with_scaler(features)

        test_ds = self.tf_dataset(features_scaled, labels, shuffle=False)
        return test_ds


    def get_input_shape(self):
        if self.input_shape is None:
            raise ValueError("Input shape is not set yet. Call split_data or prepare_train_val first.")
        return self.input_shape


# later i will write this code better
# generic preprocess, pre-split transform, post-splits transform