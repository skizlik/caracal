# caracal/data.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any, Union

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataHandler(ABC):
    """
    An abstract base class for a data handler.

    This class defines a generic interface for loading, splitting, and preparing data.
    Each concrete implementation returns different data formats - check the return_format
    property to understand what format the load() method will return.
    """

    def __init__(self, data_path: str):
        """
        Initialize the data handler with a path to the data source.

        Args:
            data_path (str): Path to the data source (file or directory)
        """
        self.data_path = data_path
        self._validate_data_path()

    def _validate_data_path(self):
        """Validate that the data path exists."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    @abstractmethod
    def load(self, **kwargs) -> Tuple[Any, ...]:
        """
        Loads and prepares the dataset.

        Returns:
            Tuple containing train, validation, and test splits.
            The exact format depends on the concrete implementation.
            Check the return_format property for details.
        """
        pass

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Return a string identifier for the data type this handler processes."""
        pass

    @property
    @abstractmethod
    def return_format(self) -> str:
        """
        Return a string describing the format of data returned by load().

        Possible values:
        - "tf_datasets": Returns (tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset])
        - "split_arrays": Returns (X_train, X_val, X_test, y_train, y_val, y_test) as arrays
        - "generators": Returns (TimeseriesGenerator, TimeseriesGenerator, TimeseriesGenerator)
        """
        pass


class ImageDataHandler(DataHandler):
    """
    A concrete data handler for image datasets.

    This class is specialized for datasets with a directory structure where
    folder names are class labels.

    Returns: (train_dataset, val_dataset, test_dataset) as tf.data.Dataset objects
    """

    @property
    def data_type(self) -> str:
        return "image"

    @property
    def return_format(self) -> str:
        return "tf_datasets"

    def __init__(self, data_path: str, image_size: Tuple[int, int], batch_size: int = 32, seed: int = 42):
        super().__init__(data_path)

        # Validate that this is a directory
        if not os.path.isdir(self.data_path):
            raise ValueError("ImageDataHandler requires a directory path with class subdirectories")

        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

        # Discover class names from subdirectories
        self.class_names = sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ])

        if not self.class_names:
            raise ValueError(f"No class subdirectories found in {self.data_path}")

    def load(self, validation_split: float = 0.2, test_split: float = 0.1) -> Tuple[
        tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
        """
        Loads and splits the dataset into training, validation, and test sets.

        Args:
            validation_split (float): Fraction of data to use for validation
            test_split (float): Fraction of data to use for testing

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
                (train_dataset, validation_dataset, test_dataset)
        """
        # Validate split ratios
        if validation_split + test_split >= 1.0:
            raise ValueError("Sum of validation_split and test_split must be < 1.0")

        # Count total files for information
        total_files = sum([
            len(os.listdir(os.path.join(self.data_path, d)))
            for d in self.class_names
        ])

        val_size = int(total_files * validation_split)
        test_size = int(total_files * test_split)
        train_size = total_files - val_size - test_size

        print(f"Total files found: {total_files}")
        print(f"Training set size: ~{train_size}")
        print(f"Validation set size: ~{val_size}")
        print(f"Test set size: ~{test_size}")

        # Create the full dataset
        full_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size
        )

        # Calculate number of batches for each set
        total_batches = tf.data.experimental.cardinality(full_ds).numpy()
        train_batches = int(total_batches * (1 - validation_split - test_split))
        val_batches = int(total_batches * validation_split)

        # Split the dataset using take() and skip()
        train_ds = full_ds.take(train_batches)
        remaining_ds = full_ds.skip(train_batches)

        if test_split > 0:
            val_ds = remaining_ds.take(val_batches)
            test_ds = remaining_ds.skip(val_batches)
        else:
            val_ds = remaining_ds
            test_ds = None

        # Pre-fetch and cache for performance
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        if test_ds is not None:
            test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def plot_sample(self, dataset: tf.data.Dataset, num_samples: int = 9):
        """Plots a random sample of images from a dataset."""
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(min(num_samples, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.show()


class TabularDataHandler(DataHandler):
    """
    A concrete data handler for structured, tabular data.

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test) as pandas DataFrames and Series
    """

    @property
    def data_type(self) -> str:
        return "tabular"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(self, data_path: str, target_column: str, features: Optional[List[str]] = None):
        super().__init__(data_path)

        # Validate that this is a file
        if not os.path.isfile(self.data_path):
            raise ValueError("TabularDataHandler requires a file path (CSV)")

        self.target_column = target_column
        self.features = features
        self.data: Optional[pd.DataFrame] = None

    def load(self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Loads data from a CSV file and splits it into train, validation, and test sets.

        Args:
            test_split (float): Fraction of data to use for testing
            val_split (float): Fraction of remaining data to use for validation
            random_state (int): Random seed for reproducible splits

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
                (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Validate split ratios
        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        # Load the data
        self.data = pd.read_csv(self.data_path)

        # Validate target column exists
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")

        # Define features and target
        if self.features:
            missing_features = set(self.features) - set(self.data.columns)
            if missing_features:
                raise ValueError(f"Features not found in data: {missing_features}")
            X = self.data[self.features]
        else:
            X = self.data.drop(columns=[self.target_column])

        y = self.data[self.target_column]

        print(f"Loaded tabular data: {len(self.data)} rows, {len(X.columns)} features")
        print(f"Target column: {self.target_column}")

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )

        # Split the training set into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_split, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


class TextDataHandler(DataHandler):
    """
    A concrete data handler for text data.

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test) as numpy arrays
    """

    @property
    def data_type(self) -> str:
        return "text"

    @property
    def return_format(self) -> str:
        return "split_arrays"

    def __init__(self, data_path: str, max_words: int = 10000, max_len: int = 100):
        super().__init__(data_path)

        # Validate that this is a file
        if not os.path.isfile(self.data_path):
            raise ValueError("TextDataHandler requires a file path (CSV)")

        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words)

    def load(self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads text data, tokenizes, and pads sequences.

        Args:
            test_split (float): Fraction of data to use for testing
            val_split (float): Fraction of remaining data to use for validation
            random_state (int): Random seed for reproducible splits

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Validate split ratios
        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        # Load the data (assuming CSV with 'text' and 'label' columns)
        data = pd.read_csv(self.data_path)

        # Validate required columns
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("Text data must have 'text' and 'label' columns")

        print(f"Loaded text data: {len(data)} samples")
        print(f"Vocabulary size limit: {self.max_words}")
        print(f"Sequence length limit: {self.max_len}")

        # Tokenize the text data
        self.tokenizer.fit_on_texts(data['text'])
        sequences = self.tokenizer.texts_to_sequences(data['text'])

        # Pad the sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, data['label'],
            test_size=test_split, random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_split, random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


class TimeSeriesDataHandler(DataHandler):
    """
    A concrete data handler for time series data.

    Returns: (train_generator, val_generator, test_generator) as TimeseriesGenerator objects
    """

    @property
    def data_type(self) -> str:
        return "timeseries"

    @property
    def return_format(self) -> str:
        return "generators"

    def __init__(self, data_path: str, sequence_length: int = 10):
        super().__init__(data_path)

        # Validate that this is a file
        if not os.path.isfile(self.data_path):
            raise ValueError("TimeSeriesDataHandler requires a file path (CSV)")

        self.sequence_length = sequence_length

    def load(self, test_split: float = 0.2, val_split: float = 0.1, random_state: int = 42) -> Tuple[
        TimeseriesGenerator, TimeseriesGenerator, TimeseriesGenerator]:
        """
        Loads time series data and creates time series generators for training.

        Args:
            test_split (float): Fraction of data to use for testing
            val_split (float): Fraction of remaining data to use for validation
            random_state (int): Random seed (unused for time series to preserve temporal order)

        Returns:
            Tuple[TimeseriesGenerator, TimeseriesGenerator, TimeseriesGenerator]:
                (train_generator, val_generator, test_generator)
        """
        # Validate split ratios
        if test_split + val_split >= 1.0:
            raise ValueError("Sum of test_split and val_split must be < 1.0")

        # Load the data (assuming CSV with 'value' column, ignoring random_state for temporal data)
        data_df = pd.read_csv(self.data_path)

        # Validate required columns
        if 'value' not in data_df.columns:
            raise ValueError("Time series data must have a 'value' column")

        data = data_df['value'].values.reshape(-1, 1)

        print(f"Loaded time series data: {len(data)} time points")
        print(f"Sequence length: {self.sequence_length}")

        # Split the data (preserving temporal order)
        total_len = len(data)
        train_size = int(total_len * (1 - test_split - val_split))
        val_size = int(total_len * val_split)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

        # Create time series generators for each split
        train_generator = TimeseriesGenerator(
            train_data, train_data,
            length=self.sequence_length, batch_size=32
        )
        val_generator = TimeseriesGenerator(
            val_data, val_data,
            length=self.sequence_length, batch_size=32
        )
        test_generator = TimeseriesGenerator(
            test_data, test_data,
            length=self.sequence_length, batch_size=32
        )

        return train_generator, val_generator, test_generator