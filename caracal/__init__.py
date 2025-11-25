# caracal/__init__.py
# v.0.0.4

"""
Caracal: A Machine Learning Framework for Variability and Reproducibility Analysis

Caracal provides tools for conducting systematic variability studies, statistical analysis
of model performance, experiment tracking, and comprehensive ML workflow management.
"""
# Core imports (always available)
from .config import ModelConfig
from .core import BaseModelWrapper, TENSORFLOW_AVAILABLE, SKLEARN_AVAILABLE
from .utils import load_object, save_object, train_val_test_split

# Exception classes (always available)
from .exceptions import (
    CaracalError,
    DataValidationError,
    ModelError,
    ExperimentError,
    StatisticalTestError,
    ConfigurationError
)

# Data handling (always available)
from .data import (
    DataHandler
)

# Loggers (always available - BaseLogger works without dependencies)
from .loggers import BaseLogger

# Experiment runners (always available)
from .runners import ExperimentRunner, run_variability_study, VariabilityStudyResults

# Memory management
from .memory import (
    MemoryManager,
    managed_memory,
    get_memory_manager,
    cleanup_gpu_memory,
    get_memory_info
)

# Global settings
from .settings import set_verbose, set_display_plots

# Build __all__ list with core functionality
__all__ = [
    # Core classes
    'ModelConfig',
    'BaseModelWrapper',
    'BaseLogger',

    # Data handling
    'DataHandler',

    # Experiment running
    'ExperimentRunner',
    'run_variability_study',
    'VariabilityStudyResults',

    # Exception classes
    'CaracalError',
    'DataValidationError',
    'ModelError',
    'ExperimentError',
    'StatisticalTestError',
    'ConfigurationError',

    # Utilities
    'load_object',
    'save_object',
    'train_val_test_split',

    # Memory cleanup
    'MemoryManager',
    'managed_memory',
    'get_memory_manager',
    'cleanup_gpu_memory',
    'get_memory_info',

    # session display variables
    'set_verbose',
    'set_display_plots',

    # Feature availability flags
    'TENSORFLOW_AVAILABLE',
    'SKLEARN_AVAILABLE',
    '__version__',


]

# Framework-specific model wrappers
if TENSORFLOW_AVAILABLE:
    from .core import KerasModelWrapper

    __all__.append('KerasModelWrapper')

if SKLEARN_AVAILABLE:
    from .core import ScikitLearnModelWrapper

    __all__.append('ScikitLearnModelWrapper')

# Data handlers with their dependencies
_data_handlers_loaded = []

try:
    from .data import ImageDataHandler

    __all__.append('ImageDataHandler')
    _data_handlers_loaded.append('ImageDataHandler')
except ImportError:
    pass

try:
    from .data import TabularDataHandler

    __all__.append('TabularDataHandler')
    _data_handlers_loaded.append('TabularDataHandler')
except ImportError:
    pass

try:
    from .data import TextDataHandler

    __all__.append('TextDataHandler')
    _data_handlers_loaded.append('TextDataHandler')
except ImportError:
    pass

try:
    from .data import TimeSeriesDataHandler

    __all__.append('TimeSeriesDataHandler')
    _data_handlers_loaded.append('TimeSeriesDataHandler')
except ImportError:
    pass

try:
    from .data import ArraysDataHandler
    __all__.append('ArraysDataHandler')
    _data_handlers_loaded.append('ArraysDataHandler')
except ImportError:
    pass

# Placed after handlers because it orchestrates them
try:
    from .data import auto_resolve_handler
    __all__.append('auto_resolve_handler')
except ImportError:
    pass

# Simple API
try:
    from .api import variability_study, compare_models
    __all__.extend(['variability_study', 'compare_models'])
except ImportError:
    pass

# Statistical analysis functions (enhanced with comprehensive testing)
try:
    from .analysis import (
        StatisticalTestResult,
        mann_whitney_test,
        wilcoxon_signed_rank_test,
        anova_test,
        kruskal_wallis_test,
        shapiro_wilk_test,
        compare_two_models,
        compare_multiple_models,
        generate_statistical_summary,
        create_results_dataframe,
        assess_training_stability,
        calculate_autocorr,
        calculate_averaged_autocorr,
        check_convergence,
        get_confusion_matrix_df,
        # Validation and effect size functions
        validate_sample_sizes,
        check_normality,
        check_equal_variances,
        check_independence,
        cohens_d,
        rank_biserial_correlation,
        eta_squared,
        apply_multiple_comparison_correction
    )

    __all__.extend([
        # Core statistical result class
        'StatisticalTestResult',

        # Statistical tests (enhanced)
        'mann_whitney_test',
        'wilcoxon_signed_rank_test',
        'anova_test',
        'kruskal_wallis_test',
        'shapiro_wilk_test',

        # High-level comparison functions
        'compare_two_models',
        'compare_multiple_models',

        # Result processing and reporting
        'generate_statistical_summary',
        'create_results_dataframe',

        # ML-specific analysis
        'assess_training_stability',

        # Convergence analysis
        'calculate_autocorr',
        'calculate_averaged_autocorr',
        'check_convergence',

        # Utility functions
        'get_confusion_matrix_df',

        # Advanced statistical functions
        'validate_sample_sizes',
        'check_normality',
        'check_equal_variances',
        'check_independence',
        'cohens_d',
        'rank_biserial_correlation',
        'eta_squared',
        'apply_multiple_comparison_correction'
    ])
    _has_statistical_functions = True
except ImportError:
    _has_statistical_functions = False

# Plotting functions (enhanced for new statistical results)
try:
    from .plotting import (
        plot_confusion_matrix,
        plot_training_history,
        plot_roc_curve,
        plot_precision_recall_curve,
        plot_averaged_autocorr,
        plot_variability_summary,
        plot_autocorr_vs_lag,
        plot_pacf_vs_lag,
        plot_averaged_pacf,
        plot_multiple_comparisons,
        plot_pairwise_comparison_matrix,
        plot_training_stability
    )

    __all__.extend([
        'plot_confusion_matrix',
        'plot_training_history',
        'plot_roc_curve',
        'plot_precision_recall_curve',
        'plot_averaged_autocorr',
        'plot_variability_summary',
        'plot_autocorr_vs_lag',
        'plot_pacf_vs_lag',
        'plot_averaged_pacf',
        'plot_multiple_comparisons',
        'plot_pairwise_comparison_matrix',
        'plot_training_stability'
    ])
    _has_plotting_functions = True
except ImportError as e:
    _has_plotting_functions = False
    if os.getenv('DEBUG'):
        print(f"Plotting functions not available: {e}")

# MLflow logger and utilities
try:
    from .loggers import MLflowLogger
    from .utils import setup_mlflow  # Moved from loggers to utils per our discussion

    __all__.extend(['MLflowLogger', 'setup_mlflow'])
    _has_mlflow_logger = True
except ImportError:
    _has_mlflow_logger = False

# Hyperparameter tuning
try:
    from .tuning import HyperparameterTuner, create_search_space

    __all__.extend(['HyperparameterTuner', 'create_search_space'])
    _has_hyperparameter_tuning = True
except ImportError:
    _has_hyperparameter_tuning = False

# Explainability features (enhanced)
try:
    from .explainers import (
        plot_shap_summary,
        plot_shap_waterfall,
        plot_shap_dependence,
        get_shap_feature_importance
    )

    __all__.extend([
        'plot_shap_summary',
        'plot_shap_waterfall',
        'plot_shap_dependence',
        'get_shap_feature_importance'
    ])
    _has_explainability = True
except ImportError:
    _has_explainability = False

# Library version
__version__ = "0.0.4"


# Enhanced feature availability summary
def get_feature_availability() -> dict:
    """
    Get a comprehensive summary of which optional features are available.

    Returns:
        Dict mapping feature names to availability status and details
    """
    statistical_detail = {}
    if _has_statistical_functions:
        statistical_detail = {
            'enhanced_tests': True,
            'effect_sizes': True,
            'assumption_validation': True,
            'multiple_comparisons': True,
            'high_level_comparisons': True
        }

    try:
        import joblib
        has_process_isolation = True
    except ImportError:
        has_process_isolation = False

    return {
        'tensorflow_support': TENSORFLOW_AVAILABLE,
        'sklearn_support': SKLEARN_AVAILABLE,
        'statistical_functions': _has_statistical_functions,
        'statistical_details': statistical_detail,
        'plotting_functions': _has_plotting_functions,
        'mlflow_logger': _has_mlflow_logger,
        'hyperparameter_tuning': _has_hyperparameter_tuning,
        'explainability': _has_explainability,
        'data_handlers': _data_handlers_loaded,
        'memory_management': True,
        'enhanced_memory_management': True,
        'process_isolation': has_process_isolation,
        'enhanced_experiment_results': True  # VariabilityStudyResults always available
    }


def print_feature_summary():
    """Print a comprehensive summary of available and missing features."""
    features = get_feature_availability()

    print("Caracal v0.0.2 Feature Availability")
    print("=" * 40)

    # Core framework support
    print("\nCore Framework Support:")
    print(f"  TensorFlow/Keras: {'✓' if features['tensorflow_support'] else '✗'}")
    print(f"  Scikit-learn: {'✓' if features['sklearn_support'] else '✗'}")
    print(f"  Memory Cleanup: ✓")
    print(f"  Enhanced cleanup system: ✓")
    print(f"  Process isolation: {'✓' if features['process_isolation'] else '✗ (missing: joblib)'}")

    # Data handlers
    print(f"\nData Handlers ({len(features['data_handlers'])}/4):")
    all_handlers = ['ImageDataHandler', 'TabularDataHandler', 'TextDataHandler', 'TimeSeriesDataHandler']
    for handler in all_handlers:
        status = '✓' if handler in features['data_handlers'] else '✗'
        print(f"  {handler}: {status}")

    # Enhanced statistical analysis
    print(f"\nStatistical Analysis: {'✓' if features['statistical_functions'] else '✗'}")
    if features['statistical_functions']:
        details = features['statistical_details']
        print(f"  Enhanced tests with effect sizes: {'✓' if details.get('enhanced_tests') else '✗'}")
        print(f"  Assumption validation:           {'✓' if details.get('assumption_validation') else '✗'}")
        print(f"  Multiple comparison corrections: {'✓' if details.get('multiple_comparisons') else '✗'}")
        print(f"  High-level comparison functions: {'✓' if details.get('high_level_comparisons') else '✗'}")

    # Experiment running
    print(f"\nExperiment Management:")
    print(f"  Variability studies:        ✓")
    print(f"  Enhanced result objects:    {'✓' if features['enhanced_experiment_results'] else '✗'}")
    print(f"  Statistical integration:    {'✓' if features['statistical_functions'] else '✗'}")

    # Optional features
    print("\nOptional Features:")
    optional_features = [
        ('Plotting & Visualization', 'plotting_functions', 'matplotlib, seaborn'),
        ('MLflow Integration', 'mlflow_logger', 'mlflow'),
        ('Hyperparameter Tuning', 'hyperparameter_tuning', 'hyperopt'),
        ('Model Explainability', 'explainability', 'shap')
    ]

    for name, key, deps in optional_features:
        status = '✓' if features[key] else '✗'
        if not features[key]:
            print(f"  {name}: {status} (missing: {deps})")
        else:
            print(f"  {name}: {status}")

    # Installation suggestions
    missing_deps = []
    if not features['statistical_functions']:
        missing_deps.append('scipy')
    if not features['plotting_functions']:
        missing_deps.extend(['matplotlib', 'seaborn'])
    if not features['mlflow_logger']:
        missing_deps.append('mlflow')
    if not features['hyperparameter_tuning']:
        missing_deps.append('hyperopt')
    if not features['explainability']:
        missing_deps.append('shap')
    if not features['tensorflow_support']:
        missing_deps.append('tensorflow')

    if missing_deps:
        print(f"\nTo enable all features:")
        print(f"  pip install {' '.join(set(missing_deps))}")
    else:
        print(f"\n✓ All optional features are available!")


def install_instructions():
    """Print detailed installation instructions for missing dependencies."""
    features = get_feature_availability()

    print("Caracal v0.0.2 Installation Guide")
    print("=" * 35)

    instructions = []

    if not features['tensorflow_support']:
        instructions.append({
            'feature': 'TensorFlow/Keras Support',
            'command': 'pip install tensorflow',
            'enables': ['KerasModelWrapper', 'ImageDataHandler', 'TextDataHandler', 'TimeSeriesDataHandler',
                        'neural network support']
        })

    if not features['sklearn_support']:
        instructions.append({
            'feature': 'Scikit-learn Support',
            'command': 'pip install scikit-learn',
            'enables': ['ScikitLearnModelWrapper', 'data splitting functions', 'confusion matrices']
        })

    if not features['statistical_functions']:
        instructions.append({
            'feature': 'Enhanced Statistical Analysis',
            'command': 'pip install scipy',
            'enables': ['All statistical tests with effect sizes', 'assumption validation',
                        'multiple comparison corrections', 'high-level comparison functions']
        })

    if not features['plotting_functions']:
        instructions.append({
            'feature': 'Plotting & Visualization',
            'command': 'pip install matplotlib seaborn',
            'enables': ['All plot_* functions', 'statistical result visualization', 'training history plots']
        })

    if not features['mlflow_logger']:
        instructions.append({
            'feature': 'MLflow Experiment Tracking',
            'command': 'pip install mlflow',
            'enables': ['MLflowLogger', 'setup_mlflow', 'advanced experiment tracking and artifact logging']
        })

    if not features['hyperparameter_tuning']:
        instructions.append({
            'feature': 'Hyperparameter Optimization',
            'command': 'pip install hyperopt',
            'enables': ['HyperparameterTuner', 'create_search_space', 'automated parameter optimization']
        })

    if not features['explainability']:
        instructions.append({
            'feature': 'Model Explainability & Interpretability',
            'command': 'pip install shap',
            'enables': ['All SHAP plotting functions', 'feature importance analysis', 'model interpretation tools']
        })

    if instructions:
        for instruction in instructions:
            print(f"\n{instruction['feature']}:")
            print(f"  Install: {instruction['command']}")
            print(f"  Enables: {', '.join(instruction['enables'])}")
    else:
        print("\n✓ All dependencies are installed!")

    print(f"\nRecommended full installation:")
    print(f"  pip install tensorflow scikit-learn scipy matplotlib seaborn mlflow hyperopt shap")

    print(f"\nMinimal installation (core functionality only):")
    print(f"  pip install scikit-learn scipy")


# Enhanced convenience import check functions
def require_tensorflow():
    """Raise error if TensorFlow not available."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for this functionality. Install with: pip install tensorflow")


def require_sklearn():
    """Raise error if scikit-learn not available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for this functionality. Install with: pip install scikit-learn")


def require_statistical():
    """Raise error if enhanced statistical functions not available."""
    if not _has_statistical_functions:
        raise ImportError("Enhanced statistical analysis requires scipy. Install with: pip install scipy")


def require_plotting():
    """Raise error if plotting functions not available."""
    if not _has_plotting_functions:
        raise ImportError(
            "Plotting functions require matplotlib and seaborn. Install with: pip install matplotlib seaborn")


def require_mlflow():
    """Raise error if MLflow not available."""
    if not _has_mlflow_logger:
        raise ImportError("MLflow integration requires mlflow. Install with: pip install mlflow")


def require_hyperopt():
    """Raise error if hyperparameter tuning not available."""
    if not _has_hyperparameter_tuning:
        raise ImportError("Hyperparameter tuning requires hyperopt. Install with: pip install hyperopt")


def require_shap():
    """Raise error if SHAP not available."""
    if not _has_explainability:
        raise ImportError("Explainability features require SHAP. Install with: pip install shap")


# Add requirement functions and utility functions to __all__
__all__.extend([
    'get_feature_availability',
    'print_feature_summary',
    'install_instructions',
    'require_tensorflow',
    'require_sklearn',
    'require_statistical',
    'require_plotting',
    'require_mlflow',
    'require_hyperopt',
    'require_shap'
])


# Quick start guide function
def quick_start_example():
    """Print a quick start example showing the enhanced workflow."""
    print("Caracal v0.0.2 Quick Start Example")
    print("=" * 35)
    print("""
# 1. Run a variability study
results = caracal.run_variability_study(
    model_builder=your_model_builder,
    data_handler=your_data_handler, 
    model_config=your_config
)

# 2. Extract metrics for statistical analysis
final_accuracies = results.get_final_metrics('val_accuracy')

# 3. Perform statistical comparison with effect sizes
statistical_analysis = caracal.compare_multiple_models(
    final_accuracies,
    correction_method='holm'
)

# 4. Generate comprehensive summary
summary = caracal.generate_statistical_summary([
    statistical_analysis['overall_test']
] + list(statistical_analysis['pairwise_comparisons'].values()))

print(summary)

# 5. View study summary
print(results.summarize())
""")


__all__.append('quick_start_example')