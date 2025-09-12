# caracal/runners.py

import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Callable, Dict, Any, List, Optional, Tuple, Union

from .core import BaseModelWrapper
from .config import ModelConfig
from .loggers import BaseLogger


class ExperimentRunner:
    """
    An engine for running a variability study across multiple model fits.

    This class orchestrates the training of a model multiple times to assess
    the stability and variability of its performance.
    """

    def __init__(self, model_builder: Callable[[ModelConfig], BaseModelWrapper],
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 model_config: ModelConfig,
                 X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                 logger: BaseLogger = BaseLogger()):

        self.model_builder = model_builder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model_config = model_config
        self.logger = logger

        self.all_runs_metrics: List[pd.DataFrame] = []
        self.final_val_accuracies: List[float] = []
        self.final_test_metrics: List[Dict[str, Any]] = []

    def _run_single_fit(self, run_id: int) -> Optional[pd.DataFrame]:
        self.logger.log_params({'run_num': run_id})
        self._cleanup_gpu_memory()

        try:
            wrapped_model = self.model_builder(self.model_config)

            print(f" - Training model {run_id}/{self.model_config.num_runs}...")
            wrapped_model.fit(train_data=(self.X_train, self.y_train),
                              validation_data=(self.X_val, self.y_val),
                              epochs=self.model_config.epochs_per_run,
                              batch_size=self.model_config.batch_size,
                              verbose=self.model_config.verbose)

            if wrapped_model.history:
                history_df = pd.DataFrame(wrapped_model.history.history)
                history_df['run_num'] = run_id
                history_df['epoch'] = history_df.index + 1  # Add the 'epoch' column

                # Rename the columns for consistency in plotting
                history_df.rename(columns={
                    'accuracy': 'train_accuracy',
                    'loss': 'train_loss',
                    'val_accuracy': 'val_accuracy',
                    'val_loss': 'val_loss'
                }, inplace=True)

                final_val_acc = history_df['val_accuracy'].iloc[-1]
                self.logger.log_metric('final_val_accuracy', final_val_acc, step=run_id)

                if self.X_test is not None and self.y_test is not None:
                    test_metrics = wrapped_model.evaluate(data=(self.X_test, self.y_test))
                    self.final_test_metrics.append(test_metrics)
                    for metric_name, value in test_metrics.items():
                        self.logger.log_metric(f'final_test_{metric_name}', value, step=run_id)

                print(
                    f" - Run {run_id}/{self.model_config.num_runs} completed. Final Validation Accuracy: {final_val_acc:.4f}")
                return history_df
            else:
                print(f" - Run {run_id}/{self.model_config.num_runs} failed to produce a training history.")
                return None
        except Exception as e:
            print(f" - Run {run_id}/{self.model_config.num_runs} failed with an error: {e}")
            return None

    def _cleanup_gpu_memory(self):
        """Force GPU memory cleanup between training runs"""
        tf.keras.backend.clear_session()
        gc.collect()

        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.reset_memory_growth(gpu)
                    tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    def run_study(self) -> Tuple[List[pd.DataFrame], List[float], List[Dict[str, Any]]]:
        """Orchestrates the entire variability study."""
        print(f"Starting Variability Study for {self.model_config.num_runs} runs.")
        self.logger.log_params(self.model_config.params)

        try:
            for i in range(self.model_config.num_runs):
                metrics_df = self._run_single_fit(run_id=i + 1)
                if metrics_df is not None:
                    self.all_runs_metrics.append(metrics_df)
                    self.final_val_accuracies.append(metrics_df['val_accuracy'].iloc[-1])
        except KeyboardInterrupt:
            print(f"\nKernel interrupted. Displaying results for {len(self.all_runs_metrics)} runs completed.")
        finally:
            self.logger.end_run()
            return self.all_runs_metrics, self.final_val_accuracies, self.final_test_metrics


def run_variability_study(model_builder: Callable[[ModelConfig], BaseModelWrapper],
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          model_config: ModelConfig,
                          X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
                          logger: BaseLogger = BaseLogger()) -> Tuple[
    List[pd.DataFrame], List[float], Optional[List[Dict[str, Any]]]]:
    """
    A simple, standalone function to run a variability study.

    This acts as a low-friction functional API" entry point for users.
    """
    runner = ExperimentRunner(model_builder=model_builder,
                              X_train=X_train, y_train=y_train,
                              X_val=X_val, y_val=y_val,
                              X_test=X_test, y_test=y_test,
                              model_config=model_config,
                              logger=logger)

    all_metrics, final_accuracies, final_test_metrics = runner.run_study()

    return all_metrics, final_accuracies, final_test_metrics