# caracal/plotting.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING

# Import global settings
from . import settings

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    Figure = None
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

if TYPE_CHECKING:
    from .core import BaseModelWrapper


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")


def _check_seaborn():
    if not HAS_SEABORN:
        raise ImportError("seaborn required for plotting. Install with: pip install seaborn")


def _check_sklearn_metrics():
    if not HAS_SKLEARN_METRICS:
        raise ImportError("scikit-learn required for ROC/PR curves. Install with: pip install scikit-learn")


def _check_tensorflow_utils():
    if not HAS_TENSORFLOW_UTILS:
        raise ImportError("TensorFlow required for multi-class plotting. Install with: pip install tensorflow")


def _should_show(show_arg: Optional[bool]) -> bool:
    """Helper to determine whether to display the plot."""
    if show_arg is not None:
        return show_arg
    return settings.should_display()


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = "", show: Optional[bool] = None) -> 'Figure':
    """Plots a confusion matrix heatmap from a DataFrame."""
    _check_matplotlib()
    _check_seaborn()

    fig = plt.figure(figsize=(10, 8))
    sn.heatmap(cm_df, annot=True, fmt="d", cmap='Blues')
    plt.title(title if title else "Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if _should_show(show):
        plt.show()
    return fig


def plot_training_history(history: Any, title: str = None, metrics: List[str] = None,
                          show: Optional[bool] = None) -> 'Figure':
    """
    Plot training and validation metrics from a history object.
    """
    _check_matplotlib()

    # Convert various history formats to DataFrame
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
            return None

    # Determine metrics to plot
    if metrics is None:
        available = history_df.columns.tolist()
        default_metrics = []
        if 'accuracy' in available or 'train_accuracy' in available:
            default_metrics.append('accuracy')
        if 'loss' in available or 'train_loss' in available:
            default_metrics.append('loss')

        if default_metrics:
            metrics = default_metrics
        else:
            metrics = [col for col in available if not col.startswith('val_') and col != 'epoch']
            if not metrics:
                print("No metrics found to plot.")
                return None

    if title is None:
        n_epochs = len(history_df)
        title = f'Training Progress: {n_epochs} Epochs'

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    train_color = '#1f77b4'
    val_color = '#ff7f0e'

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Handle column naming
        if metric in history_df.columns:
            train_col = metric
        elif f'train_{metric}' in history_df.columns:
            train_col = f'train_{metric}'
        else:
            continue

        val_col = f'val_{metric}'
        metric_display = metric.replace('_', ' ').title()
        if metric.lower() in ['mse', 'mae', 'rmse']:
            metric_display = metric.upper()
        elif metric.lower() == 'auc':
            metric_display = 'AUC'

        epochs = range(1, len(history_df) + 1)

        ax.plot(epochs, history_df[train_col], label=f'Training', color=train_color, linewidth=2)

        if val_col in history_df.columns:
            ax.plot(epochs, history_df[val_col], label=f'Validation', color=val_color, linewidth=2)
            final_train = history_df[train_col].iloc[-1]
            final_val = history_df[val_col].iloc[-1]
            ax.plot([], [], ' ', label=f'Final: {final_train:.4f} / {final_val:.4f}')
        else:
            final_train = history_df[train_col].iloc[-1]
            ax.plot([], [], ' ', label=f'Final: {final_train:.4f}')

        ax.set_title(metric_display, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_display, fontsize=11)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if any(x in metric.lower() for x in ['accuracy', 'auc', 'precision', 'recall', 'f1']):
            ax.set_ylim([0, 1.05])

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if _should_show(show):
        plt.show()
    return fig


def plot_roc_curve(model_wrapper: 'BaseModelWrapper', X_test: np.ndarray, y_test: np.ndarray,
                   title: str = "", show: Optional[bool] = None) -> 'Figure':
    """Plots the ROC curve for a multi-class model."""
    _check_matplotlib()
    _check_sklearn_metrics()
    _check_tensorflow_utils()

    if not hasattr(model_wrapper, 'predict_proba'):
        print("Model does not have a predict_proba method. Cannot plot ROC curve.")
        return None

    y_score = model_wrapper.predict_proba(X_test)

    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    num_classes = y_score.shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    fig = plt.figure(figsize=(10, 8))

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

    if _should_show(show):
        plt.show()
    return fig


def plot_precision_recall_curve(model_wrapper: 'BaseModelWrapper', X_test: np.ndarray, y_test: np.ndarray,
                                title: str = "", show: Optional[bool] = None) -> 'Figure':
    """Plots the Precision-Recall curve for a multi-class model."""
    _check_matplotlib()
    _check_sklearn_metrics()
    _check_tensorflow_utils()

    if not hasattr(model_wrapper, 'predict_proba'):
        print("Model does not have a predict_proba method. Cannot plot Precision-Recall curve.")
        return None

    y_score = model_wrapper.predict_proba(X_test)

    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T

    num_classes = y_score.shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    fig = plt.figure(figsize=(10, 8))

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

    if _should_show(show):
        plt.show()
    return fig


def plot_variability_summary(all_runs_metrics_list: List[pd.DataFrame],
                             final_metrics_series: Union[pd.Series, List],
                             final_test_series: Optional[Union[pd.Series, List]] = None,
                             metric: str = 'accuracy',
                             train_color: str = '#1f77b4',
                             val_color: str = '#ff7f0e',
                             show_histogram: bool = True,
                             show_boxplot: bool = False,
                             show_mean_lines: bool = True,
                             show: Optional[bool] = None) -> 'Figure':
    """Create a comprehensive visualization of a variability study's results."""
    _check_matplotlib()
    _check_seaborn()

    if not all_runs_metrics_list:
        print("No metrics provided for plotting.")
        return None

    def to_series(data, metric_key):
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.Series([metrics.get(metric_key) for metrics in data])
        return pd.Series(data)

    base_metric = metric.replace('train_', '').replace('val_', '')
    train_col = f'train_{base_metric}'
    val_col = f'val_{base_metric}'

    if all_runs_metrics_list:
        first_run = all_runs_metrics_list[0]
        if base_metric in first_run.columns and train_col not in first_run.columns:
            train_col = base_metric

    bounded_0_1 = any(x in base_metric.lower() for x in ['accuracy', 'auc', 'precision', 'recall', 'f1'])

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

    # 1. Main Variability Plot
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
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if n_runs <= 5:
        alpha = 0.8
    elif n_runs <= 20:
        alpha = 0.4
    else:
        alpha = max(0.15, 1.0 / n_runs * 3)

    for i, run_data in enumerate(all_runs_metrics_list):
        epochs = run_data['epoch'] if 'epoch' in run_data.columns else range(1, len(run_data) + 1)

        if i == 0:
            if train_col in run_data.columns:
                ax.plot(epochs, run_data[train_col], alpha=alpha, color=train_color, label='Training')
            if val_col in run_data.columns:
                ax.plot(epochs, run_data[val_col], alpha=alpha, color=val_color, label='Validation')
        else:
            if train_col in run_data.columns:
                ax.plot(epochs, run_data[train_col], alpha=alpha, color=train_color)
            if val_col in run_data.columns:
                ax.plot(epochs, run_data[val_col], alpha=alpha, color=val_color)

    if show_mean_lines and n_runs > 1:
        train_runs = [run[train_col].values for run in all_runs_metrics_list if train_col in run.columns]
        val_runs = [run[val_col].values for run in all_runs_metrics_list if val_col in run.columns]

        if train_runs:
            train_mean = np.mean(train_runs, axis=0)
            epochs_array = range(1, len(train_mean) + 1)
            ax.plot(epochs_array, train_mean, color=train_color, linewidth=2.5, linestyle='-', label='Mean Training',
                    zorder=10)

        if val_runs:
            val_mean = np.mean(val_runs, axis=0)
            epochs_array = range(1, len(val_mean) + 1)
            ax.plot(epochs_array, val_mean, color=val_color, linewidth=2.5, linestyle='-', label='Mean Validation',
                    zorder=10)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if bounded_0_1:
        ax.set_ylim([0, 1.05])

    plot_index = 1

    # 2. Histogram
    if show_histogram and not final_metrics_series.empty:
        ax = axes[plot_index]
        mean_final = np.mean(final_metrics_series)
        std_final = np.std(final_metrics_series)

        sn.histplot(y=final_metrics_series, bins='auto', kde=True, color=val_color, alpha=0.6, ax=ax, stat='density',
                    label='Validation')

        if final_test_series is not None and not final_test_series.empty:
            sn.histplot(y=final_test_series, bins='auto', kde=True, color='#2ca02c', alpha=0.6, ax=ax, stat='density',
                        label='Test')

        ax.axhline(mean_final, color=val_color, linestyle='--', linewidth=2)
        x_pos = ax.get_xlim()[1] * 0.7
        ax.text(x_pos, mean_final, f'μ={mean_final:.3f}\nσ={std_final:.3f}', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=val_color, alpha=0.8))

        ax.set_title(f'Distribution of Final {metric_display}', fontsize=12)
        ax.set_ylabel(f'Final {metric_display}', fontsize=11)
        ax.set_xlabel('Density', fontsize=11)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plot_index += 1

    # 3. Boxplot
    if show_boxplot and not final_metrics_series.empty:
        ax = axes[plot_index]
        data_to_plot = [final_metrics_series.values]
        labels = ['Validation']
        colors = [val_color]

        if final_test_series is not None and not final_test_series.empty:
            data_to_plot.append(final_test_series.values)
            labels.append('Test')
            colors.append('#2ca02c')

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(f'Final {metric_display}', fontsize=11)
        ax.set_title(f'Final {metric_display} Summary', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Variability Study: {n_runs} Runs × {n_epochs} Epochs', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if _should_show(show):
        plt.show()
    return fig


def plot_multiple_comparisons(comparison_results: Dict[str, Any],
                              figsize: Tuple[int, int] = (14, 10),
                              show_effect_sizes: bool = True,
                              show_corrected: bool = True,
                              show: Optional[bool] = None) -> 'Figure':
    """Generates a comprehensive multi-panel visualization for comparison results."""
    _check_matplotlib()
    _check_seaborn()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    overall_result = comparison_results['overall_test']

    # 1. Overall test result
    ax1 = fig.add_subplot(gs[0, 0])
    p_val = overall_result.p_value
    is_sig = overall_result.is_significant()
    color = 'crimson' if is_sig else 'steelblue'

    ax1.barh(['Overall Test'], [p_val], color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axvline(x=0.05, color='red', linestyle='--', alpha=0.8, linewidth=2, label='α = 0.05')
    ax1.set_xlabel('P-value', fontsize=11, fontweight='bold')
    ax1.set_title(f'{overall_result.test_name}\n{"Significant" if is_sig else "Not Significant"}', fontsize=12,
                  fontweight='bold', color=color)
    ax1.legend(loc='upper right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.text(p_val, 0, f' p={p_val:.4f}', va='center', ha='left' if p_val < 0.5 else 'right', fontsize=10,
             fontweight='bold')

    # 2. Effect size visualization
    ax2 = fig.add_subplot(gs[0, 1])
    if overall_result.effect_size is not None and show_effect_sizes:
        effect_colors = {'negligible': '#d3d3d3', 'small': '#87CEEB', 'medium': '#FFA500', 'large': '#DC143C'}
        interpretation = overall_result.effect_size_interpretation
        color = effect_colors.get(interpretation, 'gray')

        ax2.barh([overall_result.effect_size_name], [overall_result.effect_size], color=color, alpha=0.8,
                 edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Effect Size', fontsize=11, fontweight='bold')
        ax2.set_title(f'Overall Effect Size\n{interpretation.title()}', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        thresholds = {'small': 0.01, 'medium': 0.06, 'large': 0.14}
        if 'eta' in overall_result.effect_size_name.lower():
            for label, thresh in thresholds.items():
                ax2.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
                ax2.text(thresh, -0.3, label, ha='center', fontsize=8, style='italic')
    else:
        ax2.text(0.5, 0.5, 'Effect size not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')

    # 3. Summary statistics
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    summary_lines = [
        "Statistical Summary", "=" * 30,
        f"Test: {overall_result.test_name}",
        f"Statistic: {overall_result.statistic:.3f}",
        f"P-value: {overall_result.p_value:.4f}",
        f"Significance: {'Yes ✓' if is_sig else 'No ✗'}", ""
    ]
    if overall_result.effect_size is not None:
        summary_lines.extend([
            f"Effect Size:",
            f"  {overall_result.effect_size_name}: {overall_result.effect_size:.3f}",
            f"  Interpretation: {overall_result.effect_size_interpretation}", ""
        ])
    if 'pairwise_comparisons' in comparison_results:
        n_comparisons = len(comparison_results['pairwise_comparisons'])
        n_significant = len(comparison_results.get('significant_comparisons', []))
        pct = f"{n_significant / n_comparisons * 100:.1f}%" if n_comparisons > 0 else "N/A"
        summary_lines.extend([
            f"Pairwise Comparisons:",
            f"  Total: {n_comparisons}",
            f"  Significant: {n_significant} ({pct})",
            f"  Correction: {comparison_results.get('correction_method', 'none')}"
        ])
    ax3.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax3.transAxes, fontfamily='monospace', fontsize=10,
             va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

    # 4. Assumptions check
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    if overall_result.assumptions_met:
        assumption_lines = ["Assumption Checks", "=" * 30]
        for assumption, met in overall_result.assumptions_met.items():
            status = "✓" if met else "✗"
            assumption_lines.append(f"{status} {assumption.replace('_', ' ').title()}")
        if overall_result.warnings:
            assumption_lines.extend(["", "Warnings:"])
            for warning in overall_result.warnings[:3]:
                assumption_lines.append(f"  ⚠ {warning[:50]}...")
        ax4.text(0.05, 0.95, '\n'.join(assumption_lines), transform=ax4.transAxes, fontfamily='monospace', fontsize=9,
                 va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

    # 5. Pairwise comparisons
    if 'pairwise_comparisons' in comparison_results:
        ax5 = fig.add_subplot(gs[2, :])
        pairwise = comparison_results['pairwise_comparisons']
        comparison_names = list(pairwise.keys())
        display_names = [name.replace('_vs_', '\nvs\n') for name in comparison_names]
        p_values = [result.p_value for result in pairwise.values()]
        corrected_p_values = [result.corrected_p_value or result.p_value for result in pairwise.values()]
        significant = [result.is_significant() for result in pairwise.values()]

        x = np.arange(len(comparison_names))
        width = 0.35
        bars1 = ax5.bar(x - width / 2, p_values, width, label='Original p-value', alpha=0.7, edgecolor='black',
                        linewidth=0.5)
        bars2 = ax5.bar(x + width / 2, corrected_p_values, width, label=f'Corrected p-value', alpha=0.7,
                        edgecolor='black', linewidth=0.5)

        for bar, sig in zip(bars2, significant):
            bar.set_color('crimson' if sig else 'steelblue')

        ax5.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, linewidth=2, label='α = 0.05', zorder=0)
        ax5.set_xlabel('Pairwise Comparisons', fontsize=11, fontweight='bold')
        ax5.set_ylabel('P-value (log scale)', fontsize=11, fontweight='bold')
        ax5.set_title(
            f'Pairwise Comparisons with {comparison_results.get("correction_method", "unknown").title()} Correction',
            fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(display_names, rotation=0, ha='center', fontsize=9)
        ax5.legend(loc='upper right')
        ax5.set_yscale('log')
        ax5.grid(axis='y', alpha=0.3, which='both')

        for i, (corr_p, sig) in enumerate(zip(corrected_p_values, significant)):
            if sig:
                y_pos = corr_p * 1.2
                ax5.text(i, y_pos, '***' if corr_p < 0.001 else '**' if corr_p < 0.01 else '*', ha='center',
                         va='bottom', fontsize=14, fontweight='bold', color='crimson')

    plt.suptitle('Multiple Comparisons Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if _should_show(show):
        plt.show()
    return fig


def plot_pairwise_comparison_matrix(comparison_results: Dict[str, Any],
                                    figsize: Tuple[int, int] = (12, 10),
                                    show_effect_sizes: bool = True,
                                    annotate_significance: bool = True,
                                    show: Optional[bool] = None) -> 'Figure':
    """Creates a matrix visualization of pairwise comparison results."""
    _check_matplotlib()
    _check_seaborn()

    if 'pairwise_comparisons' not in comparison_results:
        print("No pairwise comparisons to plot.")
        return None

    pairwise = comparison_results['pairwise_comparisons']
    model_names = set()
    for comp_name in pairwise.keys():
        names = comp_name.split('_vs_')
        model_names.update(names)
    model_names = sorted(list(model_names))
    n_models = len(model_names)

    p_value_matrix = np.ones((n_models, n_models))
    significance_matrix = np.zeros((n_models, n_models))
    effect_size_matrix = np.zeros((n_models, n_models))

    for comp_name, result in pairwise.items():
        name1, name2 = comp_name.split('_vs_')
        i, j = model_names.index(name1), model_names.index(name2)
        p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value
        p_value_matrix[i, j] = p_val
        p_value_matrix[j, i] = p_val
        if result.is_significant():
            significance_matrix[i, j] = 1
            significance_matrix[j, i] = 1
        if result.effect_size is not None:
            effect_size_matrix[i, j] = result.effect_size
            effect_size_matrix[j, i] = result.effect_size

    n_plots = 2 if not show_effect_sizes or not np.any(effect_size_matrix) else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1: axes = [axes]

    def format_annotation(data, i, j):
        val = data[i, j]
        if i == j: return ""
        text = f'{val:.3f}'
        if annotate_significance and significance_matrix[i, j] == 1:
            if val < 0.001:
                text += '\n***'
            elif val < 0.01:
                text += '\n**'
            elif val < 0.05:
                text += '\n*'
        return text

    annot_matrix = np.array(
        [[format_annotation(p_value_matrix, i, j) for j in range(n_models)] for i in range(n_models)])
    mask = np.eye(n_models, dtype=bool)

    sn.heatmap(p_value_matrix, xticklabels=model_names, yticklabels=model_names, annot=annot_matrix, fmt='',
               cmap='RdYlGn_r', vmin=0, vmax=0.1, ax=axes[0], mask=mask, cbar_kws={'label': 'P-value'}, linewidths=0.5,
               linecolor='gray')
    axes[0].set_title(f'Corrected P-values', fontsize=12, fontweight='bold')

    sig_annot = np.array(
        [['✓' if significance_matrix[i, j] and i != j else '' for j in range(n_models)] for i in range(n_models)])
    sn.heatmap(significance_matrix, xticklabels=model_names, yticklabels=model_names, annot=sig_annot, fmt='',
               cmap='RdYlGn', vmin=0, vmax=1, ax=axes[1], mask=mask, cbar_kws={'label': 'Significant', 'ticks': [0, 1]},
               linewidths=0.5, linecolor='gray')
    axes[1].set_title('Significant Differences (α = 0.05)', fontsize=12, fontweight='bold')

    if show_effect_sizes and np.any(effect_size_matrix):
        effect_annot = np.array(
            [[f'{effect_size_matrix[i, j]:.3f}' if i != j else '' for j in range(n_models)] for i in range(n_models)])
        sn.heatmap(np.abs(effect_size_matrix), xticklabels=model_names, yticklabels=model_names, annot=effect_annot,
                   fmt='',
                   cmap='YlOrRd', ax=axes[2], mask=mask, cbar_kws={'label': 'Effect Size (absolute)'}, linewidths=0.5,
                   linecolor='gray')
        axes[2].set_title('Effect Sizes', fontsize=12, fontweight='bold')

    plt.suptitle(f'Pairwise Comparison Matrix ({n_models} models)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if _should_show(show):
        plt.show()
    return fig


def plot_training_stability(stability_results: Dict[str, Any], figsize: Tuple[int, int] = (12, 8),
                            show: Optional[bool] = None) -> 'Figure':
    """Visualizes training stability metrics."""
    _check_matplotlib()

    if 'error' in stability_results:
        print(f"Cannot plot training stability: {stability_results['error']}")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Final loss distribution
    ax1 = axes[0, 0]
    if 'final_losses_list' in stability_results:
        final_losses = stability_results['final_losses_list']
        mean_loss = stability_results['final_loss_mean']
        std_loss = stability_results['final_loss_std']
        n_runs = stability_results['n_runs']
        ax1.hist(final_losses, bins=min(10, n_runs // 2 + 1), alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(mean_loss, color='red', linestyle='--', label=f'Mean: {mean_loss:.4f}')
        ax1.axvline(mean_loss - std_loss, color='orange', linestyle=':', alpha=0.7)
        ax1.axvline(mean_loss + std_loss, color='orange', linestyle=':', alpha=0.7)
        ax1.legend()
    ax1.set_xlabel('Final Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Final Loss Distribution')

    # 2. Stability metrics
    ax2 = axes[0, 1]
    ax2.axis('off')
    stability_text = [
        "Training Stability Metrics", "=" * 25,
        f"Runs analyzed: {stability_results['n_runs']}",
        f"Epochs per run: {stability_results['common_length']}", "",
        f"Final Loss:",
        f"  Mean: {stability_results['final_loss_mean']:.4f}",
        f"  Std:  {stability_results['final_loss_std']:.4f}",
        f"  CV:   {stability_results['final_loss_cv']:.4f}", "",
        f"Stability: {stability_results['stability_assessment'].upper()}",
        f"Convergence rate: {stability_results['convergence_rate']:.1%}"
    ]
    ax2.text(0.05, 0.95, '\n'.join(stability_text), transform=ax2.transAxes, fontfamily='monospace', fontsize=10,
             va='top')

    # 3. Convergence status
    ax3 = axes[1, 0]
    converged = stability_results['converged_runs']
    not_converged = stability_results['n_runs'] - converged
    ax3.pie([converged, not_converged], labels=['Converged', 'Not Converged'], colors=['green', 'red'],
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('Convergence Analysis')

    # 4. Stability level
    ax4 = axes[1, 1]
    stability_levels = ['High', 'Moderate', 'Low']
    current_stability = stability_results['stability_assessment'].title()
    colors = ['green' if level == current_stability else 'lightgray' for level in stability_levels]
    bars = ax4.bar(stability_levels, [1, 1, 1], color=colors, alpha=0.7)
    if current_stability in stability_levels:
        bars[stability_levels.index(current_stability)].set_height(1.2)
        bars[stability_levels.index(current_stability)].set_alpha(1.0)
    ax4.set_ylabel('Stability Level')
    ax4.set_title(f'Overall Assessment: {current_stability}')
    ax4.set_ylim(0, 1.5)

    plt.tight_layout()

    if _should_show(show):
        plt.show()
    return fig


def plot_autocorr_vs_lag(data: Union[pd.Series, List[float]], max_lag: int = 20,
                         title: str = "Autocorrelation of Loss", show: Optional[bool] = None) -> 'Figure':
    """Plots the autocorrelation of a time series as a function of lag."""
    _check_matplotlib()
    if not isinstance(data, pd.Series): data = pd.Series(data)
    if len(data) <= max_lag: return None

    autocorr_values = [data.autocorr(lag) for lag in range(1, max_lag + 1)]
    lags = range(1, max_lag + 1)

    fig = plt.figure(figsize=(10, 6))
    plt.stem(lags, autocorr_values)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)

    if _should_show(show):
        plt.show()
    return fig


def plot_averaged_autocorr(lags: List[float], mean_autocorr: List[float], std_autocorr: List[float],
                           title: str = "Averaged Autocorrelation of Loss", show: Optional[bool] = None) -> 'Figure':
    """Plots the mean autocorrelation across multiple runs."""
    _check_matplotlib()
    fig = plt.figure(figsize=(10, 6))
    plt.plot(lags, mean_autocorr, 'b-', label='Mean Autocorrelation')
    plt.fill_between(lags, np.array(mean_autocorr) - np.array(std_autocorr),
                     np.array(mean_autocorr) + np.array(std_autocorr), color='b', alpha=0.2, label='Standard Deviation')
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.grid(True)

    if _should_show(show):
        plt.show()
    return fig


def plot_pacf_vs_lag(data: Union[pd.Series, List[float]], max_lag: int = 20,
                     title: str = "Partial Autocorrelation of Loss", alpha: float = 0.05,
                     show: Optional[bool] = None) -> 'Figure':
    """Plots PACF with confidence intervals."""
    _check_matplotlib()
    try:
        from statsmodels.tsa.stattools import pacf
    except ImportError:
        print("statsmodels required for PACF.")
        return None

    if not isinstance(data, pd.Series): data = pd.Series(data)
    if len(data) <= max_lag + 1: return None

    pacf_values, conf_int = pacf(data, nlags=max_lag, alpha=alpha)
    pacf_values = pacf_values[1:]
    conf_int = conf_int[1:]
    lags = range(1, max_lag + 1)

    fig = plt.figure(figsize=(10, 6))
    plt.stem(lags, pacf_values, use_line_collection=True, label='PACF')
    plt.fill_between(lags, conf_int[:, 0] - pacf_values, conf_int[:, 1] - pacf_values, alpha=0.2, color='gray')
    plt.title(title)
    plt.grid(True)

    if _should_show(show):
        plt.show()
    return fig


def plot_averaged_pacf(lags: List[float], mean_pacf: List[float], std_pacf: List[float],
                       title: str = "Averaged Partial Autocorrelation of Loss", conf_level: float = 0.95,
                       show: Optional[bool] = None) -> 'Figure':
    """Plots mean PACF across multiple runs."""
    _check_matplotlib()
    fig = plt.figure(figsize=(10, 6))
    plt.plot(lags, mean_pacf, 'b-', label='Mean PACF', linewidth=2)
    plt.fill_between(lags, np.array(mean_pacf) - np.array(std_pacf),
                     np.array(mean_pacf) + np.array(std_pacf), color='b', alpha=0.2, label='±1 Standard Deviation')

    n_points = len(lags) * 10
    conf_bound = 1.96 / np.sqrt(n_points)
    plt.axhline(y=conf_bound, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=-conf_bound, color='gray', linestyle=':', alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.grid(True)

    if _should_show(show):
        plt.show()
    return fig