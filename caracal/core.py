# caracal/core.py
import pandas as pd
import numpy as np
import gc
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from .memory import MemoryManager, managed_memory_context

# Optional TensorFlow imports
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

# Optional scikit-learn imports
try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    BaseEstimator = None
    SKLEARN_AVAILABLE = False

# Abstract Base Class for ModelWrappers

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
        self._memory_manager = MemoryManager(enable_monitoring=True)

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
        try:
            self._cleanup_implementation()
            cleanup_results = self._memory_manager.cleanup_all(force=True)

            # log significant cleanup events
            memory_freed = cleanup_results.get('memory_freed_mb', 0)
            if memory_freed >= 50:
                print(f"Model cleanup freed {memory_freed:.0f}MB")

            # report cleanup failures
            cleanup_results_dict = cleanup_results.get('cleanup_results', {})
            failed_cleaners = [name for name, result in cleanup_results_dict.items()
                                if not result.get('success', True)]
            if failed_cleaners:
                print(f"Warning: Some cleanup operations failed: {failed_cleaners}")

        except Exception as e:
            print(f"Cleanup warning: {e}")




    @abstractmethod
    def _cleanup_implementation(self):
        """Framework-specific cleanup implementation."""
        pass

    @abstractmethod
    def fit(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs):
        """
        Fits the encapsulated model.
        The type of `train_data` and `validation_data` depends on the
        concrete implementation (e.g., (X, y) tuple or tf.data.Dataset).
        """
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
    def evaluate(self, data: Any, **kwargs) -> Dict[str, Any]:
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

    @abstractmethod
    def _cleanup_implementation(self):
        """Framework-specific cleanup implementation (must be implemented by subclasses)."""
        pass

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report (available on all model wrappers)."""
        return self._memory_manager.get_memory_report()

    def check_memory_and_cleanup_if_needed(self) -> Optional[Dict[str, Any]]:
        """Check memory usage and cleanup if thresholds exceeded."""
        return self._memory_manager.check_and_cleanup_if_needed()

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
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass

        def _cleanup_model_references(self):
            """Clear model references."""
            if hasattr(self, 'model'):
                del self.model

        def _cleanup_tensorflow_session(self):
            """Clear TensorFlow session and memory."""
            tf.keras.backend.clear_session()
            gc.collect()

        def fit(self, train_data: Any, validation_data: Optional[Any] = None, **kwargs):
            if isinstance(train_data, (tf.data.Dataset, Sequence)):
                self.history = self.model.fit(train_data, validation_data=validation_data, **kwargs)
            elif isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data
                self.history = self.model.fit(x=X_train, y=y_train, validation_data=validation_data, **kwargs)
            else:
                raise TypeError(
                    "train_data must be a tuple of (X, y), a tf.data.Dataset, or a Sequence."
                )

        # REVISION: Improved prediction logic to handle binary and multi-class models correctly
        def predict(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            Generates predictions and stores them.

            For classification models, returns class indices as integers.
            For regression models, returns predictions as floats.

            Args:
                data: Input data for prediction
                **kwargs: Additional arguments passed to model.predict()

            Returns:
                np.ndarray: Predicted class labels (classification) or values (regression)
            """
            raw_predictions = self.model.predict(data, **kwargs)

            # Handle edge cases
            if np.any(np.isnan(raw_predictions)):
                raise ValueError("Model returned NaN predictions. Check your input data and model.")

            # Ensure 2D shape for consistent handling
            if raw_predictions.ndim == 1:
                raw_predictions = raw_predictions.reshape(-1, 1)

            n_samples, n_outputs = raw_predictions.shape

            # Determine if this is classification or regression
            is_classification = self._is_classification_model()

            if not is_classification:
                # Regression: return raw predictions, maintain original shape
                self.predictions = raw_predictions.flatten() if n_outputs == 1 else raw_predictions
            else:
                # Classification: return class predictions as integers
                if n_outputs == 1:
                    # Binary classification with single output (sigmoid)
                    if not np.all((raw_predictions >= 0) & (raw_predictions <= 1)):
                        raise ValueError("Binary classification model output not in [0,1] range. "
                                         "Ensure final layer uses sigmoid activation.")
                    self.predictions = (raw_predictions.flatten() > 0.5).astype(int)
                else:
                    # Multi-class classification (softmax)
                    self.predictions = np.argmax(raw_predictions, axis=1).astype(int)

            return self.predictions

        def predict_proba(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            Generates probability scores for each class.

            Only works for classification models.

            Args:
                data: Input data for prediction
                **kwargs: Additional arguments passed to model.predict()

            Returns:
                np.ndarray: Class probabilities with shape (n_samples, n_classes)
                           Always returns 2D array, even for binary classification
            """
            if not self._is_classification_model():
                raise ValueError("predict_proba() is only available for classification models. "
                                 "This appears to be a regression model.")

            raw_predictions = self.model.predict(data, **kwargs)

            # Handle edge cases
            if np.any(np.isnan(raw_predictions)):
                raise ValueError("Model returned NaN predictions. Check your input data and model.")

            # Ensure 2D shape
            if raw_predictions.ndim == 1:
                raw_predictions = raw_predictions.reshape(-1, 1)

            n_samples, n_outputs = raw_predictions.shape

            if n_outputs == 1:
                # Binary classification with sigmoid output
                prob_positive = raw_predictions.flatten()

                # Validate probabilities are in valid range
                if not np.all((prob_positive >= 0) & (prob_positive <= 1)):
                    raise ValueError("Binary classification model output not in [0,1] range. "
                                     "Ensure final layer uses sigmoid activation.")

                prob_negative = 1.0 - prob_positive
                return np.column_stack([prob_negative, prob_positive])
            else:
                # Multi-class with softmax output
                # Validate probabilities sum to approximately 1
                row_sums = np.sum(raw_predictions, axis=1)
                if not np.allclose(row_sums, 1.0, rtol=1e-3):
                    raise ValueError("Multi-class model output doesn't sum to 1 across classes. "
                                     "Ensure final layer uses softmax activation.")

                return raw_predictions

        def _is_classification_model(self) -> bool:
            """
            Determine if this is a classification model by examining the loss function.

            Returns:
                bool: True if classification, False if regression
            """
            try:
                # Try to get the loss function
                if not hasattr(self.model, 'loss') or self.model.loss is None:
                    # Model not compiled, assume classification (most common in this library)
                    return True

                loss = self.model.loss

                # Handle different loss function representations
                if hasattr(loss, '__name__'):
                    loss_name = loss.__name__.lower()
                elif hasattr(loss, 'name'):
                    loss_name = loss.name.lower()
                elif isinstance(loss, str):
                    loss_name = loss.lower()
                else:
                    loss_name = str(loss).lower()

                # Classification loss functions (comprehensive list)
                classification_indicators = [
                    'categorical_crossentropy', 'sparse_categorical_crossentropy',
                    'binary_crossentropy', 'categorical_hinge', 'sparse_categorical_hinge',
                    'hinge', 'squared_hinge', 'focal_loss', 'crossentropy'
                ]

                # Regression loss functions
                regression_indicators = [
                    'mean_squared_error', 'mse', 'mean_absolute_error', 'mae',
                    'mean_absolute_percentage_error', 'mape', 'huber_loss', 'huber',
                    'log_cosh', 'logcosh'
                ]

                # Check for regression first (more specific)
                if any(reg_loss in loss_name for reg_loss in regression_indicators):
                    return False

                # Check for classification
                if any(cls_loss in loss_name for cls_loss in classification_indicators):
                    return True

                # If we can't determine, assume classification (safer default for this library)
                return True

            except (AttributeError, TypeError):
                # Fallback: assume classification
                return True

        def _get_num_classes(self) -> int:
            """
            Helper method to determine the number of classes from model output.

            Returns:
                int: Number of classes the model predicts
            """
            try:
                output_shape = self.model.output_shape
                if isinstance(output_shape, tuple):
                    # Single output
                    return 2 if output_shape[-1] == 1 else output_shape[-1]
                elif isinstance(output_shape, list):
                    # Multiple outputs - use first output
                    return 2 if output_shape[0][-1] == 1 else output_shape[0][-1]
                else:
                    # Fallback
                    return 2
            except (AttributeError, IndexError):
                # If we can't determine from model structure, assume binary
                return 2

        def evaluate(self, data: Any, **kwargs) -> Dict[str, Any]:
            if isinstance(data, (tf.data.Dataset, Sequence)):
                results = self.model.evaluate(data, **kwargs)
            elif isinstance(data, tuple) and len(data) == 2:
                X_test, y_test = data
                results = self.model.evaluate(X_test, y_test, **kwargs)
            else:
                raise TypeError(
                    "Evaluation data must be a tuple of (X, y), a tf.data.Dataset, or a Sequence."
                )
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
            import gc
            gc.collect()

        def fit(self, train_data: Union[Tuple[np.ndarray, np.ndarray], Any],
                validation_data: Optional[Any] = None,
                epochs: Optional[int] = None,  # Accept but ignore
                batch_size: Optional[int] = None,  # Accept but ignore
                verbose: Optional[int] = None,  # Accept but ignore
                **kwargs):
            """
            Fit the scikit-learn model and create a mock history for ExperimentRunner compatibility.

            Args:
                train_data: Tuple of (X_train, y_train)
                validation_data: Optional tuple of (X_val, y_val)
                epochs: Ignored for sklearn models (Keras compatibility)
                batch_size: Ignored for sklearn models (Keras compatibility)
                verbose: Ignored for sklearn models (Keras compatibility)
                **kwargs: Additional fit parameters passed to sklearn model
            """
            if isinstance(train_data, tuple) and len(train_data) == 2:
                X_train, y_train = train_data

                # Filter out Keras-specific kwargs that sklearn doesn't understand
                sklearn_kwargs = {}
                for key, value in kwargs.items():
                    # Only pass through parameters that sklearn models typically accept
                    if key in ['sample_weight', 'check_input', 'X_idx_sorted']:
                        sklearn_kwargs[key] = value
                    # Silently ignore other parameters (like validation_split, callbacks, etc.)

                # Fit the model (sklearn models ignore epochs/batch_size/verbose from signature)
                self.model.fit(X_train, y_train, **sklearn_kwargs)

                # Calculate training accuracy
                train_accuracy = self.model.score(X_train, y_train)

                # Create mock history dictionary (sklearn trains in one "epoch")
                history_dict = {
                    'accuracy': [train_accuracy],  # Training accuracy
                    'loss': [1.0 - train_accuracy]  # Mock loss as 1 - accuracy
                }

                # Add validation metrics if validation data provided
                if validation_data is not None and isinstance(validation_data, tuple) and len(validation_data) == 2:
                    X_val, y_val = validation_data
                    val_accuracy = self.model.score(X_val, y_val)
                    history_dict['val_accuracy'] = [val_accuracy]
                    history_dict['val_loss'] = [1.0 - val_accuracy]
                else:
                    # ExperimentRunner expects validation metrics - use training metrics as fallback
                    history_dict['val_accuracy'] = [train_accuracy]
                    history_dict['val_loss'] = [1.0 - train_accuracy]

                # Create a mock history object that mimics Keras History
                class MockHistory:
                    """Mock Keras History object for sklearn compatibility."""

                    def __init__(self, history_dict):
                        self.history = history_dict
                        self.params = {}  # Empty params dict like Keras

                self.history = MockHistory(history_dict)

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