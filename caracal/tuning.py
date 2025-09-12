# caracal/tuning.py

import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from typing import Dict, Any, Callable
import warnings
import tensorflow as tf

from .core import BaseModelWrapper
from .config import ModelConfig
from .runners import ExperimentRunner

class HyperparameterTuner:
    """
    A class to automate hyperparameter tuning for a given model.
    """
    def __init__(self, model_builder: Callable, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, model_config: ModelConfig):
        self.model_builder = model_builder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_config = model_config
        self.trials = Trials()

    def tune(self, param_space: Dict[str, Any], max_evals: int = 100):
        """
        Runs hyperparameter tuning using hyperopt.
        
        Args:
            param_space (Dict): The hyperparameter search space.
            max_evals (int): The number of hyperparameter combinations to try.
        """
        def objective(params: Dict[str, Any]):
            # Suppress TensorFlow warnings inside the objective function
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Clear session to prevent memory leaks
                tf.keras.backend.clear_session()
                
                # Update config with new hyperparameters
                self.model_config.merge(params)
                
                # Build a new model with the updated config
                wrapped_model = self.model_builder(self.model_config)
                
                # Fit the model
                history = wrapped_model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), verbose=0)
                
                # Get the final validation loss as the optimization metric
                final_val_loss = history.history['val_loss'][-1]
                
            return {'loss': final_val_loss, 'status': STATUS_OK, 'history': history.history}

        # Run the tuning process
        best_params = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials
        )
        
        return best_params, self.trials
