# caracal/__init__.py

# Import core classes and functions from sub-modules
from .analysis import (
    anova_test,
    kruskal_wallis_test,
    mann_whitney_test,
    wilcoxon_signed_rank_test,
    shapiro_wilk_test
)
from .config import ModelConfig
from .core import BaseModelWrapper, TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE
from .data import DataHandler, ImageDataHandler, TabularDataHandler, TextDataHandler, TimeSeriesDataHandler
from .explainers import plot_shap_summary
from .loggers import BaseLogger
from .plotting import (
    plot_confusion_matrix,
    plot_training_history,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_averaged_autocorr,
    plot_variability_roc_curves,
    plot_variability_pr_curves
)
from .runners import ExperimentRunner
from .tuning import HyperparameterTuner
from .utils import load_object, save_object, train_val_test_split

# Conditionally import framework-specific wrappers
if TENSORFLOW_AVAILABLE:
    from .core import KerasModelWrapper
else:
    KerasModelWrapper = None

if SKLEARN_AVAILABLE:
    from .core import ScikitLearnModelWrapper
else:
    ScikitLearnModelWrapper = None

# Define the library version
__version__ = "0.0.1"

# Build __all__ list dynamically based on available dependencies
__all__ = [
    'anova_test',
    'kruskal_wallis_test',
    'mann_whitney_test',
    'wilcoxon_signed_rank_test',
    'shapiro_wilk_test',
    'ModelConfig',
    'BaseModelWrapper',
    'DataHandler', 'ImageDataHandler', 'TabularDataHandler', 'TextDataHandler', 'TimeSeriesDataHandler',
    'plot_shap_summary',
    'BaseLogger',
    'plot_confusion_matrix',
    'plot_training_history',
    'plot_roc_curve', 'plot_precision_recall_curve',
    'plot_averaged_autocorr',
    'plot_variability_roc_curves', 'plot_variability_pr_curves',
    'ExperimentRunner',
    'HyperparameterTuner',
    'load_object', 'save_object', 'train_val_test_split',
    'TENSORFLOW_AVAILABLE', 'SKLEARN_AVAILABLE',
    '__version__',
]

# Add framework-specific classes to __all__ if available
if KerasModelWrapper is not None:
    __all__.append('KerasModelWrapper')

if ScikitLearnModelWrapper is not None:
    __all__.append('ScikitLearnModelWrapper')