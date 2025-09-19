import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class Preprocessor:
    def __init__(self, cfg):
        self.random_state = cfg.seed
        self.batch_size = cfg.batch_size
        self.val_size = cfg.data.preprocessing.val_size
        self.target_col = cfg.data.target_col
        self.scaler = None
        self.encoder = cfg.data.preprocessing.encode_categorical or {"M": 1, "B": 0}
        self.feature_columns = None
        self.input_shape = None

    def load_data(self, filepath):
        return pd.read_csv(filepath)

    def split_data(self, df):

        # pre-split preprocessing
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()
        labels = df[self.target_col].map(self.encoder)
        features = df.drop(columns=[self.target_col])
        self.feature_columns = features.columns.tolist()
        self.input_shape = (features.shape[1],)


        # train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=self.val_size,
            stratify=labels,
            random_state=self.random_state
        )
        return X_train, X_val, y_train, y_val

    def fit_scaler(self, X_train):
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
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
        
        # laod data
        df = self.load_data(filepath)

        # pre-split transforms and then test/val/split
        X_train, X_val, y_train, y_val = self.split_data(df)
        
        # scaler. only fit with train
        X_train_scaled = self.fit_scaler(X_train)
        X_val_scaled = self.transform_with_scaler(X_val)

        # tf datasets
        train_ds = self.tf_dataset(X_train_scaled, y_train, shuffle=True)
        val_ds = self.tf_dataset(X_val_scaled, y_val, shuffle=False)

        # Return datasets + objects needed for MLflow logging / later reuse
        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "scaler": self.scaler,
            "encoder": self.encoder,
            "feature_columns": self.feature_columns,
            "input_shape": self.input_shape
        }

    def prepare_test(self, filepath, scaler, encoder):

        df = self.load_data(filepath)
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()
        labels = df[self.target_col].map(encoder)
        features = df.drop(columns=[self.target_col])


        
        
        #features_scaled = scaler.transform(features)
        features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)


        print(features_scaled)
        

        test_ds = self.tf_dataset(features_scaled, labels, shuffle=False)
        return test_ds


        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Run prepare_train_val first.")

        df = self.load_data(filepath)
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()
        labels = df[self.target_col].map(self.encoder)
        features = df.drop(columns=[self.target_col])
        features_scaled = self.transform_with_scaler(features)
        test_ds = self.tf_dataset(features_scaled, labels, shuffle=False)
        return test_ds
        """

    def get_input_shape(self):
        if self.input_shape is None:
            raise ValueError("Input shape is not set yet. Call split_data or prepare_train_val first.")
        return self.input_shape


