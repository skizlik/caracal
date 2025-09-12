# caracal/core.py

import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

# Check for optional dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.utils import Sequence
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    KerasModel = None
    Sequence = None
    TENSORFLOW_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    BaseEstimator = None
    SKLEARN_AVAILABLE = False


# Define the base class
class BaseModelWrapper(ABC):
    """
    A base class for wrapping a machine learning model.
    It provides a standardized interface for common analysis tasks.
    """

    def __init__(self, model: Any, model_id: str = ""):
        self.model = model
        self.model_id = model_id
        self.history: Optional[Any] = None
        self.predictions: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        model_type = self.model.__class__.__name__
        is_trained = "Yes" if self.history is not None else "No"
        return f"Caracal BaseModelWrapper(id='{self.model_id}', type='{model_type}', is_trained={is_trained})"

    def __del__(self):
        """Automatic cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass

    def cleanup(self):
        """
        Clean up resources used by this model.

        This method can be called explicitly to free resources
        before the object is destroyed.
        """
        self._cleanup_implementation()

    @abstractmethod
    def _cleanup_implementation(self):
        """Framework-specific cleanup implementation."""
        pass

    @abstractmethod
    def fit(self, train_data: Union[Tuple[np.ndarray, np.ndarray], Any], **kwargs):
        """Fits the encapsulated model."""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Generates predictions and stores them."""
        pass

    @abstractmethod
    def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Generates probability scores for each class."""
        pass

    @abstractmethod
    def evaluate(self, data: Union[Tuple[np.ndarray, np.ndarray], Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluates the encapsulated model on a given dataset.
        Returns a dictionary of metrics.
        """
        pass

    @abstractmethod
    def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Provides a quick, high-level assessment of the model's performance.
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Saves the encapsulated model."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str) -> 'BaseModelWrapper':
        """Loads a model from a file."""
        pass


# --- CONCRETE IMPLEMENTATIONS ---

# Concrete class for TensorFlow/Keras models
if TENSORFLOW_AVAILABLE:
    class KerasModelWrapper(BaseModelWrapper):
        def __init__(self, model: KerasModel, model_id: str = ""):
            super().__init__(model, model_id)
            self.history: Optional[tf.keras.callbacks.History] = None

        def _cleanup_implementation(self):
            """TensorFlow/Keras specific cleanup."""
            self._cleanup_model_references()
            self._cleanup_tensorflow_session()

        def _cleanup_model_references(self):
            """Clear model references."""
            if hasattr(self, 'model'):
                del self.model

        def _cleanup_tensorflow_session(self):
            """Clear TensorFlow session and memory."""
            tf.keras.backend.clear_session()
            gc.collect()

        def fit(self, train_data: Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset, Sequence],
                validation_data: Optional[Any] = None, **kwargs):
            if isinstance(train_data, (tf.data.Dataset, Sequence)):
                self.history = self.model.fit(train_data, validation_data=validation_data, **kwargs)
            else:  # Assumes it's a tuple of (X, y)
                X_train, y_train = train_data
                # We explicitly unpack the tuple here
                self.history = self.model.fit(x=X_train, y=y_train, validation_data=validation_data, **kwargs)

        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            raw_predictions = self.model.predict(data, **kwargs)
            if raw_predictions.ndim > 1 and raw_predictions.shape[1] > 1:
                self.predictions = np.argmax(raw_predictions, axis=1)
            else:
                self.predictions = (raw_predictions > 0.5).astype(int)
            return self.predictions

        def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
            return self.model.predict(data, **kwargs)

        def evaluate(self, data: Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset], **kwargs) -> Dict[str, Any]:
            X_test, y_test = data
            results = self.model.evaluate(X_test, y_test, **kwargs)
            metric_names = self.model.metrics_names
            return dict(zip(metric_names, results))

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")
            accuracy = accuracy_score(true_labels, self.predictions)
            return {'accuracy': accuracy}

        def save_model(self, path: str):
            self.model.save(path)
            print(f"Model saved to {path}")

        @classmethod
        def load_model(cls, path: str) -> 'KerasModelWrapper':
            loaded_model = tf.keras.models.load_model(path)
            return cls(loaded_model)

else:
    KerasModelWrapper = None


# Concrete class for Scikit-learn models
if SKLEARN_AVAILABLE:
    class ScikitLearnModelWrapper(BaseModelWrapper):
        def __init__(self, model: BaseEstimator, model_id: str = ""):
            super().__init__(model, model_id)

        def _cleanup_implementation(self):
            """Scikit-learn specific cleanup (minimal)."""
            pass

        def fit(self, train_data: Union[Tuple[np.ndarray, np.ndarray], Any], **kwargs):
            if isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data
                self.model.fit(X_train, y_train, **kwargs)
            else:
                raise ValueError("ScikitLearnModelWrapper.fit() expects train_data as a tuple of (X_train, y_train)")

        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            self.predictions = self.model.predict(data, **kwargs)
            return self.predictions

        def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(data, **kwargs)
            else:
                raise NotImplementedError("The scikit-learn model does not have a predict_proba method.")

        def evaluate(self, data: Union[Tuple[np.ndarray, np.ndarray], Any], **kwargs) -> Dict[str, Any]:
            X_test, y_test = data
            results = self.model.score(X_test, y_test, **kwargs)
            return {'accuracy': results}

        def assess(self, true_labels: np.ndarray) -> Dict[str, float]:
            if self.predictions is None:
                raise ValueError("Model has not generated predictions yet. Call predict() first.")
            accuracy = accuracy_score(true_labels, self.predictions)
            return {'accuracy': accuracy}

        def save_model(self, path: str):
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {path}")

        @classmethod
        def load_model(cls, path: str) -> 'ScikitLearnModelWrapper':
            with open(path, 'rb') as f:
                loaded_model = pickle.load(f)
            return cls(loaded_model)

else:
    ScikitLearnModelWrapper = None