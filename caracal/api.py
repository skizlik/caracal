# caracal/api.py
"""
The High-Level API for Caracal.

This module provides a interface for running variability studies
and model comparisons. It abstracts away the complexity of DataHandlers,
ModelConfigs, and ExperimentRunners into single function calls.
"""

from typing import Union, Callable, Any, Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from .core import BaseModelWrapper
from .config import ModelConfig
from .data import auto_resolve_handler, DataHandler
from .runners import run_variability_study as _run_study_internal
from .runners import VariabilityStudyResults
from .loggers import BaseLogger
from .analysis import compare_multiple_models as _stat_compare
from . import settings

def variability_study(
    model: Union[Callable, Any],
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 5,
    epochs: int = 10,
    batch_size: int = 32,
    tracker: Optional[BaseLogger] = None,
    use_process_isolation: bool = False,
    **kwargs
) -> VariabilityStudyResults:
    """
    Run a variability study on a single model with one function call.

    This function automatically detects your data format, wraps your model,
    configures the experiment, and executes the study.

    Args:
        model: The model to test. Can be:
               1. A function that returns a compiled model (Recommended)
               2. An uninstantiated class (e.g. RandomForestClassifier)
               3. An instance (will be cloned if possible)
        data: The input data. Can be:
              - "path/to/data.csv"
              - "path/to/images_dir/"
              - pd.DataFrame
              - (X, y) tuple of numpy arrays
        target_column: Name of the label column (Required for CSV/DataFrame).
        runs: Number of times to repeat the training (default: 5).
        epochs: Number of training epochs per run (default: 10).
        batch_size: Batch size for training (default: 32).
        tracker: Optional tracker (e.g., MLflowLogger) to record metrics.
        use_process_isolation: If True, runs each training in a separate process.
                               Recommended for GPU memory management.
        **kwargs: Additional arguments passed to the DataHandler
                  (e.g., image_size=(64,64), text_column="text") OR
                  passed to the ModelConfig (e.g. learning_rate=0.01).

    Returns:
        VariabilityStudyResults: Object containing all metrics and history.
        Use .summarize(), .plot(), or .to_dataframe() on this object.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> results = caracal.variability_study(
        ...     model=RandomForestClassifier,
        ...     data="dataset.csv",
        ...     target_column="label",
        ...     runs=10
        ... )
        >>> print(results.summarize())
    """
    # 1. Resolve Data Handler
    # We pass kwargs here so users can pass 'image_size', etc.
    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)

    # 2. Wrap Model Builder
    builder = _wrap_model_builder(model)

    # 3. Create Config
    # Start with explicit args
    config_dict = {
        'epochs': epochs,
        'batch_size': batch_size,
        'verbose': 1 if settings._VERBOSE else 0
    }
    
    # Extract known config keys from kwargs to pass to ModelConfig
    # (Everything else is assumed to be for the DataHandler)
    known_config_keys = [
        'learning_rate', 'optimizer', 'input_shape', 
        'n_estimators', 'max_depth', 'dropout_rate'
    ]
    for k in known_config_keys:
        if k in kwargs:
            config_dict[k] = kwargs[k]

    config = ModelConfig(config_dict)

    # 4. Run Study
    return _run_study_internal(
        model_builder=builder,
        data_handler=handler,
        model_config=config,
        num_runs=runs,
        epochs_per_run=epochs,
        tracker=tracker,
        use_process_isolation=use_process_isolation,
        gpu_memory_limit=kwargs.get('gpu_memory_limit')
    )


def compare_models(
    models: List[Union[Callable, Any]],
    data: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    target_column: Optional[str] = None,
    runs: int = 5,
    epochs: int = 10,
    metric: str = 'val_accuracy',
    use_process_isolation: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple model architectures using rigorous statistical testing.

    This function runs a variability study for EACH model provided, gathers
    the results, and performs an All-vs-All statistical comparison.

    Args:
        models: List of models to compare. Can be functions, classes, or instances.
                e.g. [RandomForestClassifier, create_cnn_model]
        data: Input data (path, DataFrame, or arrays).
        target_column: Label column name.
        runs: Number of runs per model.
        epochs: Number of epochs per run.
        metric: Metric to compare (e.g., 'val_accuracy', 'val_loss').
        use_process_isolation: Recommended True if comparing heavy Deep Learning models.
        **kwargs: Passed to DataHandler and ModelConfig.

    Returns:
        Dict containing:
            - 'overall_test': StatisticalTestResult (Kruskal-Wallis)
            - 'pairwise_comparisons': Dict of pairwise test results
            - 'summary': Text summary
    
    Example:
        >>> results = caracal.compare_models(
        ...     models=[model_A, model_B, model_C],
        ...     data=df,
        ...     target_column='target'
        ... )
        >>> caracal.plot_multiple_comparisons(results)
    """
    
    # 1. Resolve Data Handler (Once for all models)
    handler = auto_resolve_handler(data, target_column=target_column, **kwargs)
    
    results_store = {}
    
    # 2. Run Study for each model
    for i, model_input in enumerate(models):
        # Determine a readable name for logging/keys
        if hasattr(model_input, '__name__'):
            name = model_input.__name__
        elif hasattr(model_input, '__class__'):
            name = model_input.__class__.__name__
        else:
            name = f"Model_{i+1}"
            
        settings.logger.info(f"--- Evaluating {name} ---")
        
        # Wrap and Run
        # We reuse the logic from variability_study by calling it directly,
        # but we pass the pre-resolved handler to save overhead.
        
        # Build config locally
        config_dict = {
            'epochs': epochs,
            'batch_size': kwargs.get('batch_size', 32),
            'verbose': 0 # Keep individual runs quiet to avoid clutter
        }
        config = ModelConfig(config_dict)
        
        builder = _wrap_model_builder(model_input)
        
        study = _run_study_internal(
            model_builder=builder,
            data_handler=handler,
            model_config=config,
            num_runs=runs,
            epochs_per_run=epochs,
            use_process_isolation=use_process_isolation,
            gpu_memory_limit=kwargs.get('gpu_memory_limit')
        )
        
        # Store the specific metric for this model
        # We extract the raw list of final values
        final_metrics = study.get_final_metrics(metric)
        if not final_metrics:
             # Fallback for test environment or fast failures
             settings.logger.warning(f"No metrics found for {name}")
             results_store[name] = pd.Series([])
        else:
             results_store[name] = pd.Series(list(final_metrics.values()))

    # 3. Statistical Analysis
    settings.logger.info("--- Performing Statistical Comparison ---")
    
    # Filter out failed models
    valid_results = {k: v for k, v in results_store.items() if not v.empty}
    
    if len(valid_results) < 2:
        settings.logger.error("Insufficient valid results for comparison.")
        return {'error': 'Insufficient data'}

    stats_results = _stat_compare(valid_results)
    
    return stats_results


# --- Internal Helpers ---

def _wrap_model_builder(model: Any) -> Callable[[ModelConfig], BaseModelWrapper]:
    """
    Helper to convert various model inputs into a standard builder function.
    """
    # Case 1: It's already a function (Ideal)
    if callable(model) and not isinstance(model, type):
        # We assume it returns a wrapper. If it returns a raw model, we wrap it.
        def wrapper(conf):
            # Handle functions that might not accept arguments
            try:
                result = model(conf)
            except TypeError:
                result = model()
                
            if isinstance(result, BaseModelWrapper):
                return result
            return _auto_wrap_raw_model(result)
        return wrapper

    # Case 2: It's a Class (e.g. RandomForestClassifier) - Instantiate it
    if isinstance(model, type):
        def wrapper(conf):
            # Instantiate
            instance = model()
            return _auto_wrap_raw_model(instance)
        return wrapper

    # Case 3: It's an Instance - Clone it
    if hasattr(model, 'fit'):
        def wrapper(conf):
            # Try Scikit-learn clone
            try:
                from sklearn.base import clone
                new_instance = clone(model)
            except Exception:
                # Fallback for others: use as-is but warn about weight persistence
                # Note: Ideally we would deepcopy, but TF objects don't pickle well.
                settings.logger.warning(
                    "Using a pre-instantiated model instance. "
                    "Note: Weights might not reset between runs."
                )
                new_instance = model
            
            return _auto_wrap_raw_model(new_instance)
        return wrapper

    raise ValueError(f"Could not interpret model input: {model}")


def _auto_wrap_raw_model(model: Any) -> BaseModelWrapper:
    """Detects model type and wraps it in the correct Caracal wrapper."""
    model_type = str(type(model))
    
    # Check for Keras (using string check to avoid import dependency)
    if 'keras' in model_type or 'tensorflow' in model_type:
        from .core import KerasModelWrapper
        return KerasModelWrapper(model)
    
    # Check for Scikit-Learn (duck typing)
    if hasattr(model, 'fit') and hasattr(model, 'predict'):
        from .core import ScikitLearnModelWrapper
        return ScikitLearnModelWrapper(model)
    
    # TODO: Add PyTorch wrapper check here later
    
    raise TypeError(f"Could not automatically wrap model of type {type(model)}. "
                    "Please use a ModelWrapper explicitly.")
