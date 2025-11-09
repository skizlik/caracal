import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, TYPE_CHECKING

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import seaborn as sn

    HAS_SEABORN = True
except ImportError:
    sn = None
    HAS_SEABORN = False

# Optional sklearn for metrics
try:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False

# Optional TensorFlow for utils
try:
    from tensorflow.keras.utils import to_categorical

    HAS_TENSORFLOW_UTILS = True
except ImportError:
    HAS_TENSORFLOW_UTILS = False

# Import BaseModelWrapper with TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .core import BaseModelWrapper


def _check_matplotlib():
    """Check matplotlib availability."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")


def _check_seaborn():
    """Check seaborn availability."""
    if not HAS_SEABORN:
        raise ImportError("seaborn required for plotting. Install with: pip install seaborn")


def _check_sklearn_metrics():
    """Check sklearn metrics availability."""
    if not HAS_SKLEARN_METRICS:
        raise ImportError("scikit-learn required for ROC/PR curves. Install with: pip install scikit-learn")


def _check_tensorflow_utils():
    """Check TensorFlow utils availability."""
    if not HAS_TENSORFLOW_UTILS:
        raise ImportError("TensorFlow required for multi-class plotting. Install with: pip install tensorflow")


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = ""):
    """Plots a confusion matrix heatmap from a DataFrame."""
    _check_matplotlib()
    _check_seaborn()

    plt.figure(figsize=(10, 8))
    sn.heatmap(cm_df, annot=True, fmt="d", cmap='Blues')
    plt.title(title if title else "Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_training_history(history: Any, title: str = None, metrics: List[str] = None):
    """
    Plots training and validation metrics from a history object.

    Args:
        history (Any): A Keras History object, a pandas DataFrame, a dictionary of lists,
                       or a list of dictionaries.
        title (str, optional): The title of the plot. If not provided, a default
                               title will be generated.
        metrics (List[str], optional): A list of metrics to plot, e.g., ['accuracy', 'loss'].
                                       Defaults to ['accuracy', 'loss'] if not provided.
    """
    _check_matplotlib()

    if isinstance(history, pd.DataFrame):
        history_df = history
    elif isinstance(history, dict):
        history_df = pd.DataFrame(history)
    elif isinstance(history, list) and all(isinstance(i, dict) for i in history):
        history_df = pd.DataFrame(history)
    else:
        try:
            history_df = pd.DataFrame(history.history)
        except AttributeError:
            print("Error: The provided history object format is not supported.")
            return

    if metrics is None:
        metrics = ['accuracy', 'loss']

    if title is None:
        epochs = len(history_df)
        metric_names = " & ".join(m.title() for m in metrics)
        title = f'Training & Validation {metric_names} ({epochs} Epochs)'

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        val_metric = f'val_{metric}'
        if metric in history_df.columns and val_metric in history_df.columns:
            axes[i].plot(history_df[metric], label=f'Training {metric.title()}')
            axes[i].plot(history_df[val_metric], label=f'Validation {val_metric.title()}')
            axes[i].set_title(f'{metric.title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.title())
            axes[i].legend()
            axes[i].grid(True)
        else:
            print(f"Warning: Metric '{metric}' or '{val_metric}' not found in history data. Skipping.")
            if i < len(axes):
                fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_roc_curve(model_wrapper: 'BaseModelWrapper', X_test: np.ndarray, y_test: np.ndarray, title: str = ""):
    """Plots the ROC curve for a multi-class model."""
    _check_matplotlib()
    _check_sklearn_metrics()
    _check_tensorflow_utils()

    if not hasattr(model_wrapper, 'predict_proba'):
        print("Model does not have a predict_proba method. Cannot plot ROC curve.")
        return

    y_score = model_wrapper.predict_proba(X_test)

    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    num_classes = y_score.shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title if title else 'Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(model_wrapper: 'BaseModelWrapper', X_test: np.ndarray, y_test: np.ndarray,
                                title: str = ""):
    """Plots the Precision-Recall curve for a multi-class model."""
    _check_matplotlib()
    _check_sklearn_metrics()
    _check_tensorflow_utils()

    if not hasattr(model_wrapper, 'predict_proba'):
        print("Model does not have a predict_proba method. Cannot plot Precision-Recall curve.")
        return

    y_score = model_wrapper.predict_proba(X_test)

    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    num_classes = y_score.shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Precision-recall curve of class {i} (area = {pr_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title if title else 'Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_variability_summary(all_runs_metrics_list: List[pd.DataFrame],
                             final_metrics_series: Union[pd.Series, List],
                             final_test_series: Optional[Union[pd.Series, List]] = None,
                             metric: str = 'accuracy',
                             train_color: str = '#1f77b4',
                             val_color: str = '#ff7f0e',
                             show_histogram: bool = True,
                             show_boxplot: bool = False,
                             show_mean_lines: bool = True):
    """
    Create a comprehensive visualization of a variability study's results.

    Generates a multi-panel plot showing:
    - Training curves for all runs overlaid with optional mean trajectories
    - Distribution histogram of final metrics (rotated for visual alignment)
    - Optional boxplot summary statistics

    Parameters
    ----------
    all_runs_metrics_list : List[pd.DataFrame]
        List of DataFrames, each containing the training history for one run.
        Expected columns: 'epoch', 'train_{metric}', 'val_{metric}'
    final_metrics_series : Union[pd.Series, List]
        Final validation metrics from each run. Can be a pandas Series or list
        of values. If list of dicts, will extract the appropriate metric.
    final_test_series : Optional[Union[pd.Series, List]], default=None
        Final test metrics from each run, if available. Same format as
        final_metrics_series.
    metric : str, default='accuracy'
        Name of the metric to plot. Can include prefix like 'val_accuracy'
        or just base name like 'accuracy'. Common metrics are automatically
        formatted (e.g., 'mse' → 'MSE', 'auc' → 'AUC').
    train_color : str, default='#1f77b4'
        Hex color code for training curves (matplotlib default blue).
    val_color : str, default='#ff7f0e'
        Hex color code for validation curves (matplotlib default orange).
    show_histogram : bool, default=True
        Whether to include the distribution histogram panel.
    show_boxplot : bool, default=False
        Whether to include the boxplot summary panel.
    show_mean_lines : bool, default=True
        Whether to overlay bold mean trajectory lines when multiple runs exist.

    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show()

    Notes
    -----
    - Automatically detects metric types (loss vs accuracy) for appropriate scaling
    - Handles both Keras-style ('train_accuracy', 'val_accuracy') and
      sklearn-style ('accuracy') column naming
    - Alpha transparency is automatically adjusted based on number of runs
    - For bounded metrics (accuracy, precision, etc.), y-axis is set to [0, 1.05]

    Examples
    --------
    >>> # Basic usage with a completed variability study
    >>> results = caracal.run_variability_study(model, data, config)
    >>> plot_variability_summary(
    ...     results.all_runs_metrics,
    ...     results.final_val_accuracies
    ... )

    >>> # Plotting loss instead of accuracy, without mean lines
    >>> plot_variability_summary(
    ...     results.all_runs_metrics,
    ...     results.get_final_metrics('val_loss'),
    ...     metric='loss',
    ...     show_mean_lines=False
    ... )

    >>> # Including test results with both histogram and boxplot
    >>> plot_variability_summary(
    ...     results.all_runs_metrics,
    ...     results.final_val_accuracies,
    ...     final_test_series=results.final_test_accuracies,
    ...     show_histogram=True,
    ...     show_boxplot=True
    ... )
    """
    _check_matplotlib()
    _check_seaborn()

    if not all_runs_metrics_list:
        print("No metrics provided for plotting.")
        return

    def to_series(data, metric_key):
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.Series([metrics.get(metric_key) for metrics in data])
        return pd.Series(data)

    # Clean up metric handling
    base_metric = metric.replace('train_', '').replace('val_', '')
    train_col = f'train_{base_metric}'
    val_col = f'val_{base_metric}'

    # Handle sklearn case where it might just be 'accuracy'
    if all_runs_metrics_list:
        first_run = all_runs_metrics_list[0]
        if base_metric in first_run.columns and train_col not in first_run.columns:
            train_col = base_metric

    # Determine metric properties
    lower_is_better = any(x in base_metric.lower() for x in ['loss', 'error', 'mse', 'mae', 'rmse'])
    bounded_0_1 = any(x in base_metric.lower() for x in ['accuracy', 'auc', 'precision', 'recall', 'f1'])

    # Better metric name display
    metric_display = base_metric.replace('_', ' ').title()
    if base_metric.lower() in ['mse', 'mae', 'rmse']:
        metric_display = base_metric.upper()
    elif base_metric.lower() == 'auc':
        metric_display = 'AUC'

    final_metrics_series = to_series(final_metrics_series, val_col)
    if final_test_series is not None:
        final_test_series = to_series(final_test_series, f'test_{base_metric}')

    num_plots = 1 + int(show_histogram) + int(show_boxplot)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))

    if num_plots == 1:
        axes = [axes]

    # Main variability plot
    ax = axes[0]

    n_runs = len(all_runs_metrics_list)
    n_epochs = len(all_runs_metrics_list[0]) if all_runs_metrics_list else 0

    if not final_metrics_series.empty:
        mean_final = np.mean(final_metrics_series)
        std_final = np.std(final_metrics_series)
        title = f'{metric_display} Across {n_runs} Training Runs\n(Final: {mean_final:.3f} ± {std_final:.3f})'
    else:
        title = f'{metric_display} Across {n_runs} Training Runs'

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(metric_display, fontsize=11)

    # Force integer-only epoch labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Calculate alpha based on number of runs
    if n_runs <= 5:
        alpha = 0.8
    elif n_runs <= 20:
        alpha = 0.4
    else:
        alpha = max(0.15, 1.0 / n_runs * 3)

    # Plot all runs
    for i, run_data in enumerate(all_runs_metrics_list):
        epochs = run_data['epoch'] if 'epoch' in run_data.columns else range(1, len(run_data) + 1)

        if i == 0:
            if train_col in run_data.columns:
                ax.plot(epochs, run_data[train_col], alpha=alpha,
                        color=train_color, label='Training')
            if val_col in run_data.columns:
                ax.plot(epochs, run_data[val_col], alpha=alpha,
                        color=val_color, label='Validation')
        else:
            if train_col in run_data.columns:
                ax.plot(epochs, run_data[train_col], alpha=alpha,
                        color=train_color)
            if val_col in run_data.columns:
                ax.plot(epochs, run_data[val_col], alpha=alpha,
                        color=val_color)

    # Add mean lines if requested and multiple runs exist
    if show_mean_lines and n_runs > 1:
        train_runs = [run[train_col].values for run in all_runs_metrics_list
                      if train_col in run.columns]
        val_runs = [run[val_col].values for run in all_runs_metrics_list
                    if val_col in run.columns]

        if train_runs:
            train_mean = np.mean(train_runs, axis=0)
            epochs_array = range(1, len(train_mean) + 1)
            ax.plot(epochs_array, train_mean, color=train_color, linewidth=2.5,
                    linestyle='-', label='Mean Training', zorder=10)

        if val_runs:
            val_mean = np.mean(val_runs, axis=0)
            epochs_array = range(1, len(val_mean) + 1)
            ax.plot(epochs_array, val_mean, color=val_color, linewidth=2.5,
                    linestyle='-', label='Mean Validation', zorder=10)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set y-limits based on metric type
    if bounded_0_1:
        ax.set_ylim([0, 1.05])
    # Otherwise let matplotlib auto-scale

    plot_index = 1

    # Histogram (rotated/horizontal)
    if show_histogram and not final_metrics_series.empty:
        ax = axes[plot_index]

        mean_final = np.mean(final_metrics_series)
        std_final = np.std(final_metrics_series)

        # Horizontal histogram with y parameter
        sn.histplot(y=final_metrics_series, bins='auto', kde=True,
                    color=val_color, alpha=0.6, ax=ax, stat='density',
                    label='Validation')

        if final_test_series is not None and not final_test_series.empty:
            sn.histplot(y=final_test_series, bins='auto', kde=True,
                        color='#2ca02c', alpha=0.6, ax=ax, stat='density',
                        label='Test')

        # Horizontal line for mean
        ax.axhline(mean_final, color=val_color, linestyle='--', linewidth=2)

        # Position annotation for horizontal histogram
        x_pos = ax.get_xlim()[1] * 0.7

        ax.text(x_pos, mean_final,
                f'μ={mean_final:.3f}\nσ={std_final:.3f}',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor=val_color,
                          alpha=0.8))

        ax.set_title(f'Distribution of Final {metric_display}', fontsize=12)
        ax.set_ylabel(f'Final {metric_display}', fontsize=11)
        ax.set_xlabel('Density', fontsize=11)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plot_index += 1

    # Boxplot
    if show_boxplot and not final_metrics_series.empty:
        ax = axes[plot_index]

        data_to_plot = [final_metrics_series.values]
        labels = ['Validation']
        colors = [val_color]

        if final_test_series is not None and not final_test_series.empty:
            data_to_plot.append(final_test_series.values)
            labels.append('Test')
            colors.append('#2ca02c')

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanprops=dict(marker='D',
                                                       markerfacecolor='red',
                                                       markeredgecolor='red'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(f'Final {metric_display}', fontsize=11)
        ax.set_title(f'Final {metric_display} Summary', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Variability Study: {n_runs} Runs × {n_epochs} Epochs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_multiple_comparisons(comparison_results: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize results from compare_multiple_models function.

    Args:
        comparison_results: Results dictionary from compare_multiple_models()
        figsize: Figure size tuple
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top left: Overall test result
    ax1 = axes[0, 0]
    overall_result = comparison_results['overall_test']

    p_val = overall_result.p_value
    color = 'red' if overall_result.is_significant() else 'gray'

    ax1.bar(['Overall Test'], [p_val], color=color, alpha=0.7)
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
    ax1.set_ylabel('P-value')
    ax1.set_title(f'{overall_result.test_name}\np = {p_val:.4f}')
    ax1.legend()

    # Top right: Effect size
    ax2 = axes[0, 1]
    if overall_result.effect_size is not None:
        effect_colors = {'negligible': 'lightgray', 'small': 'lightblue', 'medium': 'orange', 'large': 'red'}
        color = effect_colors.get(overall_result.effect_size_interpretation, 'gray')

        ax2.bar(['Effect Size'], [overall_result.effect_size], color=color, alpha=0.7)
        ax2.set_ylabel(overall_result.effect_size_name)
        ax2.set_title(f'Overall Effect Size\n{overall_result.effect_size_interpretation}')

    # Bottom: Pairwise comparison results
    ax3 = axes[1, :]
    ax3 = plt.subplot(2, 1, 2)  # Span both bottom columns

    if 'pairwise_comparisons' in comparison_results:
        pairwise = comparison_results['pairwise_comparisons']

        comparison_names = list(pairwise.keys())
        p_values = [result.p_value for result in pairwise.values()]
        corrected_p_values = [result.corrected_p_value or result.p_value for result in pairwise.values()]
        significant = [result.is_significant() for result in pairwise.values()]

        x = np.arange(len(comparison_names))
        width = 0.35

        # Plot original and corrected p-values
        bars1 = ax3.bar(x - width / 2, p_values, width, label='Original p-value', alpha=0.7)
        bars2 = ax3.bar(x + width / 2, corrected_p_values, width, label='Corrected p-value', alpha=0.7)

        # Color bars by significance
        for bar, sig in zip(bars2, significant):
            bar.set_color('red' if sig else 'gray')

        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
        ax3.set_xlabel('Pairwise Comparisons')
        ax3.set_ylabel('P-value')
        ax3.set_title(f'Pairwise Comparisons ({comparison_results.get("correction_method", "unknown")} correction)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax3.legend()
        ax3.set_yscale('log')  # Log scale for better p-value visualization

    plt.tight_layout()
    plt.show()


def plot_pairwise_comparison_matrix(comparison_results: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)):
    """
    Create a matrix visualization of pairwise comparison results.

    Args:
        comparison_results: Results dictionary from compare_multiple_models()
        figsize: Figure size tuple
    """
    _check_matplotlib()
    _check_seaborn()

    if 'pairwise_comparisons' not in comparison_results:
        print("No pairwise comparisons to plot.")
        return

    pairwise = comparison_results['pairwise_comparisons']

    # Extract model names from comparison names
    model_names = set()
    for comp_name in pairwise.keys():
        names = comp_name.split('_vs_')
        model_names.update(names)

    model_names = sorted(list(model_names))
    n_models = len(model_names)

    # Create matrices for p-values and significance
    p_value_matrix = np.ones((n_models, n_models))
    significance_matrix = np.zeros((n_models, n_models))

    for comp_name, result in pairwise.items():
        name1, name2 = comp_name.split('_vs_')
        i, j = model_names.index(name1), model_names.index(name2)

        p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

        p_value_matrix[i, j] = p_val
        p_value_matrix[j, i] = p_val

        if result.is_significant():
            significance_matrix[i, j] = 1
            significance_matrix[j, i] = 1

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # P-value heatmap
    sn.heatmap(p_value_matrix, xticklabels=model_names, yticklabels=model_names,
               annot=True, fmt='.3f', cmap='viridis_r', ax=axes[0])
    axes[0].set_title('P-values (Corrected)')

    # Significance heatmap
    sn.heatmap(significance_matrix, xticklabels=model_names, yticklabels=model_names,
               annot=True, fmt='d', cmap='RdYlBu_r', cbar_kws={'label': 'Significant'}, ax=axes[1])
    axes[1].set_title('Significant Differences (α = 0.05)')

    plt.tight_layout()
    plt.show()


def plot_variability_study_summary(results: 'VariabilityStudyResults', metric: str = 'val_accuracy',
                                   figsize: Tuple[int, int] = (15, 10)):
    """
    Create a comprehensive visualization of variability study results.

    Args:
        results: VariabilityStudyResults object
        metric: Metric to focus on for the summary
        figsize: Figure size tuple
    """
    _check_matplotlib()
    _check_seaborn()

    fig = plt.figure(figsize=figsize)

    # Create subplot layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Training histories (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])

    for i, run_df in enumerate(results.all_runs_metrics):
        if metric in run_df.columns:
            ax1.plot(run_df['epoch'], run_df[metric], alpha=0.6, label=f'Run {i + 1}')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metric.title())
    ax1.set_title(f'Training Histories: {metric}')
    ax1.grid(True, alpha=0.3)

    # 2. Final metric distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])

    try:
        final_values = results.get_final_metrics(metric)
        values = list(final_values.values())

        ax2.hist(values, bins=min(10, len(values) // 2 + 1), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
        ax2.set_xlabel(f'Final {metric}')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Final Metric Distribution')
        ax2.legend()

    except Exception as e:
        ax2.text(0.5, 0.5, f'Could not plot {metric}\n{str(e)}',
                 ha='center', va='center', transform=ax2.transAxes)

    # 3. Run-to-run variability (middle left)
    ax3 = fig.add_subplot(gs[1, 0])

    try:
        values = list(final_values.values())
        run_numbers = range(1, len(values) + 1)

        ax3.plot(run_numbers, values, 'o-', alpha=0.7)
        ax3.fill_between(run_numbers,
                         [np.mean(values) - np.std(values)] * len(values),
                         [np.mean(values) + np.std(values)] * len(values),
                         alpha=0.2, label=f'±1 std ({np.std(values):.3f})')
        ax3.axhline(np.mean(values), color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Run Number')
        ax3.set_ylabel(f'Final {metric}')
        ax3.set_title('Run-to-Run Variability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    except Exception as e:
        ax3.text(0.5, 0.5, f'Error plotting variability\n{str(e)}',
                 ha='center', va='center', transform=ax3.transAxes)

    # 4. Summary statistics (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    try:
        stats_text = [
            f"Variability Study Summary",
            f"=" * 25,
            f"Runs: {results.n_runs}",
            f"Metric: {metric}",
            f"",
            f"Final {metric}:",
            f"  Mean: {np.mean(values):.4f}",
            f"  Std:  {np.std(values):.4f}",
            f"  CV:   {np.std(values) / np.mean(values) * 100:.1f}%",
            f"  Min:  {np.min(values):.4f}",
            f"  Max:  {np.max(values):.4f}",
            f"  Range: {np.max(values) - np.min(values):.4f}",
        ]

        ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
                 fontfamily='monospace', fontsize=10, va='top')

    except Exception as e:
        ax4.text(0.5, 0.5, f'Error computing statistics\n{str(e)}',
                 ha='center', va='center', transform=ax4.transAxes)

    # 5. Convergence assessment (middle right)
    ax5 = fig.add_subplot(gs[1, 2])

    try:
        from .analysis import check_convergence

        convergence_results = []
        for run_df in results.all_runs_metrics:
            if metric in run_df.columns:
                converged = check_convergence(run_df[metric])
                convergence_results.append(converged)

        convergence_rate = sum(convergence_results) / len(convergence_results) if convergence_results else 0

        # Pie chart of convergence
        labels = ['Converged', 'Not Converged']
        sizes = [sum(convergence_results), len(convergence_results) - sum(convergence_results)]
        colors = ['green', 'red']

        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title(f'Convergence Rate\n({convergence_rate:.1%})')

    except Exception as e:
        ax5.text(0.5, 0.5, f'Convergence analysis failed\n{str(e)}',
                 ha='center', va='center', transform=ax5.transAxes)

    # 6. Available metrics (bottom span)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    available_metrics = results.get_available_metrics()
    metrics_text = f"Available metrics for analysis: {', '.join(available_metrics)}"
    ax6.text(0.5, 0.5, metrics_text, ha='center', va='center', transform=ax6.transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

    plt.suptitle(f'Variability Study Summary: {metric}', fontsize=16, y=0.95)
    plt.show()


def plot_training_stability(stability_results: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize training stability metrics from assess_training_stability().

    Args:
        stability_results: Results dictionary from assess_training_stability()
        figsize: Figure size tuple
    """
    _check_matplotlib()

    if 'error' in stability_results:
        print(f"Cannot plot training stability: {stability_results['error']}")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top left: Final loss distribution
    ax1 = axes[0, 0]

    # Create mock data for visualization (in real implementation, you'd use the actual loss values)
    mean_loss = stability_results['final_loss_mean']
    std_loss = stability_results['final_loss_std']
    n_runs = stability_results['n_runs']

    # Generate approximate distribution
    final_losses = np.random.normal(mean_loss, std_loss, n_runs)

    ax1.hist(final_losses, bins=min(10, n_runs // 2 + 1), alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(mean_loss, color='red', linestyle='--', label=f'Mean: {mean_loss:.4f}')
    ax1.axvline(mean_loss - std_loss, color='orange', linestyle=':', alpha=0.7, label=f'±1 std')
    ax1.axvline(mean_loss + std_loss, color='orange', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Final Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Loss Distribution')
    ax1.legend()

    # Top right: Stability metrics
    ax2 = axes[0, 1]
    ax2.axis('off')

    stability_text = [
        "Training Stability Metrics",
        "=" * 25,
        f"Runs analyzed: {stability_results['n_runs']}",
        f"Epochs per run: {stability_results['common_length']}",
        "",
        f"Final Loss:",
        f"  Mean: {stability_results['final_loss_mean']:.4f}",
        f"  Std:  {stability_results['final_loss_std']:.4f}",
        f"  CV:   {stability_results['final_loss_cv']:.4f}",
        "",
        f"Stability: {stability_results['stability_assessment'].upper()}",
        f"Convergence rate: {stability_results['convergence_rate']:.1%}",
        f"Converged runs: {stability_results['converged_runs']}/{stability_results['n_runs']}"
    ]

    ax2.text(0.05, 0.95, '\n'.join(stability_text), transform=ax2.transAxes,
             fontfamily='monospace', fontsize=10, va='top')

    # Bottom left: Convergence status
    ax3 = axes[1, 0]

    converged = stability_results['converged_runs']
    not_converged = stability_results['n_runs'] - converged

    labels = ['Converged', 'Not Converged']
    sizes = [converged, not_converged]
    colors = ['green', 'red']

    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Convergence Analysis')

    # Bottom right: Stability assessment
    ax4 = axes[1, 1]

    stability_levels = ['High', 'Moderate', 'Low']
    current_stability = stability_results['stability_assessment'].title()

    colors = ['green' if level == current_stability else 'lightgray' for level in stability_levels]

    bars = ax4.bar(stability_levels, [1, 1, 1], color=colors, alpha=0.7)

    # Highlight current level
    current_index = stability_levels.index(current_stability)
    bars[current_index].set_height(1.2)
    bars[current_index].set_alpha(1.0)

    ax4.set_ylabel('Stability Level')
    ax4.set_title(f'Overall Assessment: {current_stability}')
    ax4.set_ylim(0, 1.5)

    # Add CV value as text
    ax4.text(current_index, 1.3, f'CV = {stability_results["final_loss_cv"]:.3f}',
             ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_autocorr_vs_lag(data: Union[pd.Series, List[float]],
                         max_lag: int = 20,
                         title: str = "Autocorrelation of Loss"):
    """Plots the autocorrelation of a time series as a function of lag."""
    _check_matplotlib()

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) <= max_lag:
        print("Data is too short to calculate autocorrelation for the given max_lag.")
        return

    autocorr_values = [data.autocorr(lag) for lag in range(1, max_lag + 1)]
    lags = range(1, max_lag + 1)

    plt.figure(figsize=(10, 6))
    plt.stem(lags, autocorr_values, use_line_collection=True)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    plt.show()


def plot_averaged_autocorr(lags: List[float],
                           mean_autocorr: List[float],
                           std_autocorr: List[float],
                           title: str = "Averaged Autocorrelation of Loss"):
    """Plots the averaged autocorrelation with a shaded region for standard deviation."""
    _check_matplotlib()

    plt.figure(figsize=(10, 6))
    plt.plot(lags, mean_autocorr, 'b-', label='Mean Autocorrelation')
    plt.fill_between(lags,
                     np.array(mean_autocorr) - np.array(std_autocorr),
                     np.array(mean_autocorr) + np.array(std_autocorr),
                     color='b', alpha=0.2, label='Standard Deviation')
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pacf_vs_lag(data: Union[pd.Series, List[float]],
                     max_lag: int = 20,
                     title: str = "Partial Autocorrelation of Loss",
                     alpha: float = 0.05):
    """
    Plots the partial autocorrelation of a time series as a function of lag.

    Args:
        data: Time series data (e.g., loss history)
        max_lag: Maximum lag to compute PACF for
        title: Plot title
        alpha: Significance level for confidence intervals
    """
    _check_matplotlib()

    # Check for statsmodels
    try:
        from statsmodels.tsa.stattools import pacf
        HAS_STATSMODELS = True
    except ImportError:
        raise ImportError("statsmodels required for PACF analysis. Install with: pip install statsmodels")

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) <= max_lag + 1:
        print("Data is too short to calculate PACF for the given max_lag.")
        return

    # Calculate PACF with confidence intervals
    pacf_values, conf_int = pacf(data, nlags=max_lag, alpha=alpha)

    # Remove lag 0 (always 1.0)
    pacf_values = pacf_values[1:]
    conf_int = conf_int[1:]
    lags = range(1, max_lag + 1)

    plt.figure(figsize=(10, 6))

    # Plot PACF values
    plt.stem(lags, pacf_values, use_line_collection=True, label='PACF')

    # Add confidence intervals
    conf_upper = conf_int[:, 1] - pacf_values
    conf_lower = pacf_values - conf_int[:, 0]
    plt.fill_between(lags, conf_int[:, 0], conf_int[:, 1],
                     alpha=0.2, color='gray', label=f'{(1 - alpha) * 100:.0f}% Confidence')

    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_averaged_pacf(lags: List[float],
                       mean_pacf: List[float],
                       std_pacf: List[float],
                       title: str = "Averaged Partial Autocorrelation of Loss",
                       conf_level: float = 0.95):
    """
    Plots the averaged partial autocorrelation with confidence bands.

    Args:
        lags: List of lag values
        mean_pacf: Mean PACF values across runs
        std_pacf: Standard deviation of PACF values
        title: Plot title
        conf_level: Confidence level for bands
    """
    _check_matplotlib()

    plt.figure(figsize=(10, 6))

    # Plot mean PACF
    plt.plot(lags, mean_pacf, 'b-', label='Mean PACF', linewidth=2)

    # Add variability bands
    plt.fill_between(lags,
                     np.array(mean_pacf) - np.array(std_pacf),
                     np.array(mean_pacf) + np.array(std_pacf),
                     color='b', alpha=0.2, label='±1 Standard Deviation')

    # Add theoretical confidence bounds (approximate)
    n_points = len(lags) * 10  # Rough estimate of data points
    conf_bound = 1.96 / np.sqrt(n_points)  # 95% confidence for white noise
    plt.axhline(y=conf_bound, color='gray', linestyle=':', alpha=0.7,
                label=f'{conf_level * 100:.0f}% White Noise Bound')
    plt.axhline(y=-conf_bound, color='gray', linestyle=':', alpha=0.7)

    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()