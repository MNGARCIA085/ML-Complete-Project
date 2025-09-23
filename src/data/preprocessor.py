import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class Preprocessor:
    """
    Data preprocessing utility for ML pipelines.

    This class handles:
    - Loading CSV data
    - Dropping unnecessary columns
    - Splitting into train/validation sets
    - Encoding target labels
    - Scaling features
    - Converting to TensorFlow datasets
    - Preparing test and inference-ready samples
    """

    def __init__(self, cfg):
        """
        Initialize the preprocessor with configuration settings.

        Args:
            cfg: Config object with attributes:
                - seed (int): Random seed for reproducibility.
                - batch_size (int): Batch size for training datasets.
                - data.preprocessing.val_size (float): Validation split size.
                - data.target_col (str): Target column name.
                - data.preprocessing.encode_categorical (dict): Encoding for categorical target.
        """
        self.random_state = cfg.seed
        self.batch_size = cfg.batch_size
        self.val_size = cfg.data.preprocessing.val_size
        self.target_col = cfg.data.target_col
        self.scaler = None
        self.encoder = cfg.data.preprocessing.encode_categorical or {"M": 1, "B": 0}
        self.feature_columns = None
        self.input_shape = None

    def load_data(self, filepath):
        """Load a CSV file into a pandas DataFrame."""
        return pd.read_csv(filepath)

    def split_data(self, df):
        """
        Preprocess and split the dataset into train/validation sets.

        Steps:
        - Drop ID/unnecessary columns
        - Encode target labels
        - Separate features and labels
        - Perform stratified train/validation split

        Args:
            df (pd.DataFrame): Raw dataset.

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        # Drop unwanted columns and missing values
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()

        # Encode labels
        labels = df[self.target_col].map(self.encoder)
        
        # Extract features
        features = df.drop(columns=[self.target_col])
        self.feature_columns = features.columns.tolist()
        self.input_shape = (features.shape[1],)


        # Stratified train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=self.val_size,
            stratify=labels,
            random_state=self.random_state
        )
        return X_train, X_val, y_train, y_val

    def fit_scaler(self, X_train):
        """
        Fit a StandardScaler on training features.

        Args:
            X_train (pd.DataFrame): Training features.

        Returns:
            pd.DataFrame: Scaled training features.
        """
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        return X_train_scaled

    def transform_with_scaler(self, X):
        """
        Apply fitted scaler to new features.

        Args:
            X (pd.DataFrame): Features to scale.

        Returns:
            pd.DataFrame: Scaled features.
        """
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

    def tf_dataset(self, X, y, shuffle=True, batch_size=None):
        """
        Convert features and labels into a tf.data.Dataset.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.
            shuffle (bool): Whether to shuffle dataset.
            batch_size (int, optional): Batch size (defaults to cfg.batch_size).

        Returns:
            tf.data.Dataset: TensorFlow dataset object.
        """
        bs = batch_size or self.batch_size
        ds = tf.data.Dataset.from_tensor_slices(
            (X.values.astype("float32"), y.values.astype("int32"))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        return ds.batch(bs)

    def prepare_train_val(self, filepath):
        """
        Prepare training and validation datasets from CSV file.

        Steps:
        - Load data
        - Split into train/val
        - Fit scaler on train
        - Transform val
        - Convert to tf.data.Dataset

        Args:
            filepath (str): Path to CSV file.

        Returns:
            dict: {
                "train_ds": tf.data.Dataset,
                "val_ds": tf.data.Dataset,
                "scaler": fitted scaler,
                "encoder": label encoder,
                "feature_columns": list of feature names,
                "input_shape": model input shape
            }
        """
        # Load raw data
        df = self.load_data(filepath)

        # Split into train/val
        X_train, X_val, y_train, y_val = self.split_data(df)
        
        # Scale features
        X_train_scaled = self.fit_scaler(X_train)
        X_val_scaled = self.transform_with_scaler(X_val)

        # Convert to TensorFlow datasets
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


    def prepare_sample(self, X_raw, scaler, encoder, features):
        """
        Preprocess a single sample for inference.

        Args:
            sample (dict or pd.Series): Input sample with feature names as keys.
            scaler: Trained scaler (if not passed, use self.scaler).
            encoder: Encoder mapping for target col if needed (optional).
            columns: named features

        Returns:
            np.ndarray: Preprocessed sample ready for model inference.
        """
        

        # Convert to DataFrame with proper column names
        X_df = pd.DataFrame([X_raw], columns=features) if X_raw.ndim == 1 else pd.DataFrame(X_raw, columns=features)

        # Scale
        X_scaled = scaler.transform(X_df)

        return X_scaled
        


    def prepare_test(self, filepath, scaler, encoder):
        """
        Prepare test dataset for evaluation.

        Args:
            filepath (str): Path to test CSV file.
            scaler: Fitted scaler.
            encoder (dict): Label encoder.

        Returns:
            tf.data.Dataset: Test dataset.
        """
        df = self.load_data(filepath)
    
        drop_cols = ["id", "Unnamed: 32"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()

        # Encode labels
        labels = df[self.target_col].map(encoder)

        # Scale features
        features = df.drop(columns=[self.target_col])
        features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

        # Convert to tf.data.Dataset
        test_ds = self.tf_dataset(features_scaled, labels, shuffle=False)
        return test_ds


    def get_input_shape(self):
        """
        Get input shape for the model.

        Raises:
            ValueError: If input shape has not been set yet.

        Returns:
            tuple: Input shape (n_features,)
        """
        if self.input_shape is None:
            raise ValueError("Input shape is not set yet. Call split_data or prepare_train_val first.")
        return self.input_shape




