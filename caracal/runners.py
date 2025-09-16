# caracal/runners.py

import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Callable, Dict, Any, List, Optional, Tuple, Union

from .core import BaseModelWrapper
from .config import ModelConfig
from .loggers import BaseLogger
from .data import DataHandler
from .memory import managed_memory_context

class ExperimentRunner:
    """
    An engine for running a variability study across multiple model fits.

    This class orchestrates the training of a model multiple times to assess
    the stability and variability of its performance.
    """

    def __init__(self, model_builder: Callable[[ModelConfig], BaseModelWrapper],
                 data_handler: DataHandler,
                 model_config: ModelConfig,
                 logger: BaseLogger = BaseLogger()):

        self.model_builder = model_builder
        self.data_handler = data_handler
        self.model_config = model_config
        self.logger = logger

        print("Loading and preparing data...")
        data_dict = self.data_handler.load()

        self.train_data = data_dict['train_data']
        self.val_data = data_dict.get('val_data')
        self.test_data = data_dict.get('test_data')

        self.all_runs_metrics: List[pd.DataFrame] = []
        self.final_val_accuracies: List[float] = []
        self.final_test_metrics: List[Dict[str, Any]] = []

    def _run_single_fit(self, run_id: int, epochs: int) -> Optional[pd.DataFrame]:
        """
        Run a single training iteration.

        Args:
            run_id: Identifier for this run
            epochs: Number of epochs to train for this run

        Returns:
            DataFrame with training metrics or None if failed
        """
        self.logger.log_params({'run_num': run_id})

        wrapped_model = self.model_builder(self.model_config)

        try:
            print(f" - Training model {run_id}...")
            wrapped_model.fit(train_data=self.train_data,
                              validation_data=self.val_data,
                              epochs=epochs,
                              batch_size=self.model_config.get('batch_size', 32),
                              verbose=self.model_config.get('verbose', 0))

            if wrapped_model.history:
                history_df = pd.DataFrame(wrapped_model.history.history)
                history_df['run_num'] = run_id
                history_df['epoch'] = history_df.index + 1

                history_df.rename(columns={
                    'accuracy': 'train_accuracy',
                    'loss': 'train_loss',
                    'val_accuracy': 'val_accuracy',
                    'val_loss': 'val_loss'
                }, inplace=True)

                if self.val_data is not None and 'val_accuracy' in history_df.columns:
                    final_val_acc = history_df['val_accuracy'].iloc[-1]
                    self.final_val_accuracies.append(final_val_acc)
                    self.logger.log_metric('final_val_accuracy', final_val_acc, step=run_id)
                else:
                    final_val_acc = float('nan')
                    self.logger.log_metric('final_val_accuracy', final_val_acc, step=run_id)

                if self.test_data is not None:
                    test_metrics = wrapped_model.evaluate(data=self.test_data)
                    self.final_test_metrics.append(test_metrics)
                    for metric_name, value in test_metrics.items():
                        self.logger.log_metric(f'final_test_{metric_name}', value, step=run_id)

                wrapped_model.cleanup()

                print(f" - Run {run_id} completed.")
                return history_df
            else:
                print(f" - Run {run_id} failed to produce a training history.")
                return None
        except Exception as e:
            print(f" - Run {run_id} failed with an error: {e}")
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

    def run_study(self, num_runs: int = 5, epochs_per_run: Optional[int] = None) -> Tuple[
        List[pd.DataFrame], List[float], List[Dict[str, Any]]]:
        """
        Orchestrates the entire variability study.

        Args:
            num_runs: Number of training runs to perform
            epochs_per_run: Epochs per run (defaults to config.epochs or 10)

        Returns:
            Tuple of (all_runs_metrics, final_val_accuracies, final_test_metrics)
        """
        if epochs_per_run is None:
            epochs_per_run = self.model_config.get('epochs', 10)

        print(f"Starting Variability Study for {num_runs} runs.")
        self.logger.log_params(self.model_config.params)
        self.logger.log_params({'num_runs': num_runs, 'epochs_per_run': epochs_per_run})

        try:
            for i in range(num_runs):
                metrics_df = self._run_single_fit(run_id=i + 1, epochs=epochs_per_run)
                if metrics_df is not None:
                    self.all_runs_metrics.append(metrics_df)

        except KeyboardInterrupt:
            print(f"\nKernel interrupted. Displaying results for {len(self.all_runs_metrics)} runs completed.")

        finally:
            # Final comprehensive cleanup
            print("Performing final cleanup...")
            # Use any model wrapper instance for final cleanup
            if hasattr(self, '_last_model') and self._last_model:
                final_cleanup = self._last_model.cleanup()

            self.logger.end_run()

        return self.all_runs_metrics, self.final_val_accuracies, self.final_test_metrics


class VariabilityStudyResults:
    """
    Container for variability study results with methods to easily extract data
    for statistical analysis.

    Maintains backward compatibility while providing enhanced functionality
    for integration with analysis.py functions.
    """

    def __init__(self,
                 all_runs_metrics: List[pd.DataFrame],
                 final_val_accuracies: List[float],
                 final_test_metrics: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize variability study results.

        Args:
            all_runs_metrics: List of DataFrames containing full training histories
            final_val_accuracies: List of final validation accuracies
            final_test_metrics: Optional list of test metric dictionaries
        """
        self.all_runs_metrics = all_runs_metrics
        self.final_val_accuracies = final_val_accuracies
        self.final_test_metrics = final_test_metrics or []

        self._validate_data()

    def _validate_data(self):
        """Validate that the data is consistent."""
        if not self.all_runs_metrics:
            raise ValueError("No run metrics provided")

        n_runs = len(self.all_runs_metrics)

        if len(self.final_val_accuracies) != n_runs:
            raise ValueError(f"Mismatch: {n_runs} run metrics but {len(self.final_val_accuracies)} final accuracies")

        if self.final_test_metrics and len(self.final_test_metrics) != n_runs:
            raise ValueError(f"Mismatch: {n_runs} run metrics but {len(self.final_test_metrics)} test metrics")

    def __iter__(self):
        """Support tuple unpacking for backward compatibility."""
        return iter((self.all_runs_metrics, self.final_val_accuracies, self.final_test_metrics))

    def __len__(self):
        """Return number of runs."""
        return len(self.all_runs_metrics)

    @property
    def n_runs(self) -> int:
        """Number of runs in the study."""
        return len(self.all_runs_metrics)

    def get_final_metrics(self, metric_name: str = 'val_accuracy') -> Dict[str, float]:
        """
        Extract final metric values in format ready for analysis.py functions.

        Args:
            metric_name: Name of metric to extract ('val_accuracy', 'val_loss', etc.)

        Returns:
            Dictionary mapping run names to final metric values

        Example:
            >>> results = run_variability_study(...)
            >>> final_accs = results.get_final_metrics('val_accuracy')
            >>> statistical_test = compare_multiple_models(final_accs)
        """
        if metric_name == 'val_accuracy':
            # Special case: use the pre-computed final accuracies
            values = self.final_val_accuracies
        else:
            # Extract from full training histories
            values = []
            for run_df in self.all_runs_metrics:
                if metric_name in run_df.columns:
                    values.append(run_df[metric_name].iloc[-1])
                else:
                    available_metrics = list(run_df.columns)
                    raise ValueError(f"Metric '{metric_name}' not found. Available: {available_metrics}")

        # Create dictionary with run names
        run_names = [f'run_{i + 1}' for i in range(len(values))]
        return dict(zip(run_names, values))

    def get_final_metrics_series(self, metric_name: str = 'val_accuracy') -> pd.Series:
        """
        Get final metrics as pandas Series (alternative format for some analysis functions).

        Args:
            metric_name: Name of metric to extract

        Returns:
            pandas Series with run names as index
        """
        final_dict = self.get_final_metrics(metric_name)
        return pd.Series(final_dict)

    def get_test_metrics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Extract test metrics in format ready for analysis.

        Args:
            metric_name: Name of test metric to extract

        Returns:
            Dictionary mapping run names to test metric values, or None if no test data
        """
        if not self.final_test_metrics:
            return None

        values = []
        for test_dict in self.final_test_metrics:
            if metric_name in test_dict:
                values.append(test_dict[metric_name])
            else:
                available_metrics = list(test_dict.keys())
                raise ValueError(f"Test metric '{metric_name}' not found. Available: {available_metrics}")

        run_names = [f'run_{i + 1}' for i in range(len(values))]
        return dict(zip(run_names, values))

    def get_training_histories(self, metric_name: str) -> Dict[str, pd.Series]:
        """
        Extract full training histories for a specific metric.

        Args:
            metric_name: Name of metric to extract histories for

        Returns:
            Dictionary mapping run names to training history Series

        Example:
            >>> histories = results.get_training_histories('val_loss')
            >>> autocorr_analysis = calculate_averaged_autocorr(list(histories.values()))
        """
        histories = {}

        for i, run_df in enumerate(self.all_runs_metrics):
            run_name = f'run_{i + 1}'

            if metric_name in run_df.columns:
                histories[run_name] = run_df[metric_name].copy()
            else:
                available_metrics = list(run_df.columns)
                raise ValueError(f"Metric '{metric_name}' not found in run {i + 1}. Available: {available_metrics}")

        return histories

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics across all runs."""
        all_metrics = set()

        for run_df in self.all_runs_metrics:
            all_metrics.update(run_df.columns)

        # Add test metrics if available
        if self.final_test_metrics:
            for test_dict in self.final_test_metrics:
                all_metrics.update(test_dict.keys())

        return sorted(list(all_metrics))

    def summarize(self) -> str:
        """
        Generate a text summary of the variability study results.

        Returns:
            Formatted summary string
        """
        summary_lines = [
            f"Variability Study Results",
            f"=" * 30,
            f"Number of runs: {self.n_runs}",
            f"Available metrics: {', '.join(self.get_available_metrics())}",
            "",
            f"Final Validation Accuracy:",
            f"  Mean: {np.mean(self.final_val_accuracies):.4f}",
            f"  Std:  {np.std(self.final_val_accuracies):.4f}",
            f"  Min:  {np.min(self.final_val_accuracies):.4f}",
            f"  Max:  {np.max(self.final_val_accuracies):.4f}",
        ]

        if self.final_test_metrics and 'accuracy' in self.final_test_metrics[0]:
            test_accs = [tm['accuracy'] for tm in self.final_test_metrics]
            summary_lines.extend([
                "",
                f"Final Test Accuracy:",
                f"  Mean: {np.mean(test_accs):.4f}",
                f"  Std:  {np.std(test_accs):.4f}",
                f"  Min:  {np.min(test_accs):.4f}",
                f"  Max:  {np.max(test_accs):.4f}",
            ])

        return "\n".join(summary_lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert study results to a summary DataFrame.

        Returns:
            DataFrame with one row per run and columns for final metrics
        """
        data = []

        for i in range(self.n_runs):
            row = {
                'run_id': i + 1,
                'run_name': f'run_{i + 1}',
                'final_val_accuracy': self.final_val_accuracies[i]
            }

            # Add final values of other metrics from training history
            run_df = self.all_runs_metrics[i]
            for col in run_df.columns:
                if col != 'val_accuracy':  # Avoid duplication
                    row[f'final_{col}'] = run_df[col].iloc[-1]

            # Add test metrics if available
            if i < len(self.final_test_metrics):
                test_metrics = self.final_test_metrics[i]
                for key, value in test_metrics.items():
                    row[f'test_{key}'] = value

            data.append(row)

        return pd.DataFrame(data)

    def compare_models_statistically(self, metric_name: str = 'val_accuracy',
                                     alpha: float = 0.05,
                                     correction_method: str = 'holm') -> Dict[str, Any]:
        """
        Convenience method to directly perform statistical comparison of runs.

        Args:
            metric_name: Metric to compare across runs
            alpha: Significance level
            correction_method: Multiple comparison correction method

        Returns:
            Results from compare_multiple_models()

        Note: This imports analysis functions dynamically to avoid circular imports
        """
        try:
            from .analysis import compare_multiple_models
        except ImportError:
            raise ImportError("Statistical analysis functions not available. "
                              "Ensure scipy is installed and analysis.py is working.")

        final_metrics = self.get_final_metrics(metric_name)

        return compare_multiple_models(
            final_metrics,
            alpha=alpha,
            correction_method=correction_method
        )


def run_variability_study(model_builder, data_handler, model_config,
                          num_runs: int = 5, epochs_per_run: Optional[int] = None,
                          logger=None, enable_process_isolation: bool = False) -> VariabilityStudyResults:
    """
    Enhanced variability study that returns a VariabilityStudyResults object
    with methods for easy integration with statistical analysis.

    Args:
        model_builder: Function that creates a BaseModelWrapper from ModelConfig
        data_handler: DataHandler instance
        model_config: ModelConfig with training parameters
        num_runs: Number of runs to perform (default 5)
        epochs_per_run: Epochs per run (defaults to model_config.epochs or 10)
        logger: Optional logger instance

    Returns:
        VariabilityStudyResults object with analysis methods
    """
    # Import here to avoid circular dependency
    from .loggers import BaseLogger

    if logger is None:
        logger = BaseLogger()

    runner = ExperimentRunner(
        model_builder=model_builder,
        data_handler=data_handler,
        model_config=model_config,
        logger=logger
    )

    all_metrics, final_accuracies, final_test_metrics = runner.run_study(
        num_runs=num_runs,
        epochs_per_run=epochs_per_run
    )

    # Return enhanced results object
    return VariabilityStudyResults(all_metrics, final_accuracies, final_test_metrics)