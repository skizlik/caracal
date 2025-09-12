# caracal/plotting.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from typing import Dict, Any, Optional, List, Union
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import to_categorical
from .core import BaseModelWrapper


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = ""):
    """Plots a confusion matrix heatmap from a DataFrame."""
    plt.figure(figsize=(10, 8))
    sn.heatmap(cm_df, annot=True, fmt="d", cmap='Blues')
    plt.title(title if title else "Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

        
def plot_training_history(history: Any, title: str):
    """
    Plots the training and validation accuracy and loss from a history object.
    
    Args:
        history (Any): A Keras History object or a pandas DataFrame.
        title (str): The title of the plot.
    """
    # Check if the input is a DataFrame or a Keras History object
    if isinstance(history, pd.DataFrame):
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
    else: # Assume it's a Keras History object
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model_wrapper: BaseModelWrapper, X_test: np.ndarray, y_test: np.ndarray, title: str = ""):
    """
    Plots the ROC curve for a multi-class model.
    
    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_test (np.ndarray): The testing data features.
        y_test (np.ndarray): The testing data labels.
        title (str): The title of the plot.
    """
    if isinstance(model_wrapper.model, KerasModel):
        y_score = model_wrapper.model.predict(X_test)
        
        # Binarize the true labels for plotting
        y_test_binarized = to_categorical(y_test, num_classes=y_score.shape[1])
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_score.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(y_score.shape[1]):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                         ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title if title else f'Receiver Operating Characteristic - Model: {model_wrapper.model_id}')
        plt.legend(loc="lower right")
        plt.show()

def plot_precision_recall_curve(model_wrapper: BaseModelWrapper, X_test: np.ndarray, y_test: np.ndarray, title: str = ""):
    """
    Plots the Precision-Recall curve for a multi-class model.
    
    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_test (np.ndarray): The testing data features.
        y_test (np.ndarray): The testing data labels.
        title (str): The title of the plot.
    """
    if isinstance(model_wrapper.model, KerasModel):
        y_score = model_wrapper.model.predict(X_test)
        
        # Binarize the true labels for plotting
        y_test_binarized = to_categorical(y_test, num_classes=y_score.shape[1])
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(y_score.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
            average_precision[i] = auc(recall[i], precision[i])

        plt.figure(figsize=(10, 8))
        for i in range(y_score.shape[1]):
            plt.plot(recall[i], precision[i], label='Precision-recall curve of class {0} (area = {1:0.2f})'
                                                    ''.format(i, average_precision[i]))
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title if title else f'Precision-Recall Curve - Model: {model_wrapper.model_id}')
        plt.legend(loc="lower left")
        plt.show()

def plot_variability_summary(all_runs_metrics_list: List[pd.DataFrame], final_metrics_series: pd.Series,
                             final_test_series: Optional[pd.Series] = None,
                             metric: str = 'accuracy', train_color: str = 'blue', val_color: str = 'orange',
                             show_histogram: bool = True, show_boxplot: bool = False):
    """
    Creates a composite plot of a variability study's results.

    Args:
        all_runs_metrics_list (List[pd.DataFrame]): A list of DataFrames with all epoch metrics from all runs.
        final_metrics_series (pd.Series): Series of the final metric values across all runs.
        final_test_series (Optional[pd.Series]): Series of the final test metric values.
        metric (str): The metric to plot (e.g., 'accuracy', 'loss').
        train_color (str): The color for the training metric plots.
        val_color (str): The color for the validation metric plots.
        show_histogram (bool): If True, a histogram of final metrics is plotted.
        show_boxplot (bool): If True, a box plot of final metrics is plotted.
    """
    if not all_runs_metrics_list:
        print("No metrics provided for plotting.")
        return

    num_runs = len(all_runs_metrics_list)
    num_plots = 1 + int(show_histogram) + int(show_boxplot)

    ratios = [4] * int(num_plots > 1) + [1] * (num_plots - 1)

    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(6 * num_plots, 7),
                             gridspec_kw={'width_ratios': ratios})

    if num_plots == 1:
        axes = [axes]

    # Plot the overlaid training history on the first subplot
    axes[0].set_title(f'Training and Validation {metric.title()} Across All Runs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel(metric.title())

    alphac = min(max(1.5 / num_runs, 0.1), 0.9)

    for i, run_data in enumerate(all_runs_metrics_list):
        axes[0].plot(run_data['epoch'], run_data[f'train_{metric}'],
                     alpha=alphac, linestyle='-', color=train_color, label='_nolegend_')
        axes[0].plot(run_data['epoch'], run_data[f'val_{metric}'],
                     alpha=alphac, linestyle='-', color=val_color, label='_nolegend_')

    axes[0].plot([], [], color=train_color, label=f'Training {metric.title()}')
    axes[0].plot([], [], color=val_color, label=f'Validation {metric.title()}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Conditionally plot the histogram and box plot
    plot_index = 1
    if show_histogram:
        sn.histplot(y=final_metrics_series, bins=10, kde=True, color='skyblue', ax=axes[plot_index], label='Validation')
        if final_test_series is not None:
            sn.histplot(y=final_test_series, bins=10, kde=True, color='green', ax=axes[plot_index], label='Test')

        axes[plot_index].set_title(f'Distribution of Final {metric.title()}')
        axes[plot_index].set_xlabel('Frequency of Runs')
        axes[plot_index].set_ylabel(f'Final {metric.title()}')
        axes[plot_index].grid(axis='x', linestyle='--', alpha=0.7)
        axes[plot_index].legend()
        plot_index += 1

    if show_boxplot:
        boxplot_data = {'Validation': final_metrics_series}
        if final_test_series is not None:
            boxplot_data['Test'] = final_test_series
        boxplot_df = pd.DataFrame(boxplot_data)

        sn.boxplot(data=boxplot_df, orient='v', ax=axes[plot_index])

        axes[plot_index].set_title(f'Box Plot of Final {metric.title()}')
        axes[plot_index].set_ylabel(f'Final {metric.title()}')
        axes[plot_index].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import to_categorical

from .core import BaseModelWrapper, KerasModelWrapper


def plot_variability_roc_curves(model_wrappers: List[BaseModelWrapper], X_test: np.ndarray, y_test: np.ndarray,
                                title: str = ""):
    """
    Overlays the ROC curves for multiple model runs on a single plot.

    Args:
        model_wrappers (List[BaseModelWrapper]): A list of trained ModelWrapper instances.
        X_test (np.ndarray): The testing data features.
        y_test (np.ndarray): The testing data labels.
        title (str): The title of the plot.
    """
    if not model_wrappers:
        print("No models provided for plotting.")
        return

    num_classes = model_wrappers[0].model.output_shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    plt.figure(figsize=(10, 8))

    alphac = min(max(1.5 / len(model_wrappers), 0.1), 0.9)

    for i, model_wrapper in enumerate(model_wrappers):
        if not hasattr(model_wrapper, 'predict_proba'):
            print(f"Skipping model run {i + 1}: 'predict_proba' method not found.")
            continue

        y_score = model_wrapper.predict_proba(X_test)

        # Plot ROC curve for each class
        for j in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, j], y_score[:, j])
            label_text = 'Class {}'.format(j) if i == 0 else '_nolegend_'
            plt.plot(fpr, tpr, alpha=alphac, label=label_text)

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title if title else 'Variability of ROC Curves')
    plt.legend(loc="lower right")
    plt.show()


# caracal/plotting.py

def plot_variability_pr_curves(model_wrappers: List[BaseModelWrapper], X_test: np.ndarray, y_test: np.ndarray,
                               title: str = ""):
    """
    Overlays the Precision-Recall curves for multiple model runs on a single plot.

    Args:
        model_wrappers (List[BaseModelWrapper]): A list of trained ModelWrapper instances.
        X_test (np.ndarray): The testing data features.
        y_test (np.ndarray): The testing data labels.
        title (str): The title of the plot.
    """
    if not model_wrappers:
        print("No models provided for plotting.")
        return

    num_classes = model_wrappers[0].model.output_shape[1]
    y_test_binarized = to_categorical(y_test, num_classes=num_classes)

    plt.figure(figsize=(10, 8))

    alphac = min(max(1.5 / len(model_wrappers), 0.1), 0.9)

    for i, model_wrapper in enumerate(model_wrappers):
        if not hasattr(model_wrapper, 'predict_proba'):
            print(f"Skipping model run {i + 1}: 'predict_proba' method not found.")
            continue

        y_score = model_wrapper.predict_proba(X_test)

        # Plot Precision-Recall curve for each class
        for j in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, j], y_score[:, j])
            label_text = 'Class {}'.format(j) if i == 0 else '_nolegend_'
            plt.plot(recall, precision, alpha=alphac, label=label_text)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title if title else 'Variability of Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.show()

def plot_autocorr_vs_lag(data: Union[pd.Series, List[float]], max_lag: int = 20,
                         title: str = "Autocorrelation of Loss"):
    """
    Plots the autocorrelation of a time series as a function of lag.

    Args:
        data (pd.Series or List[float]): The time series data (e.g., loss history).
        max_lag (int): The maximum lag to plot.
        title (str): The title of the plot.
    """
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


def plot_averaged_autocorr(lags: List[float], mean_autocorr: List[float], std_autocorr: List[float],
                           title: str = "Averaged Autocorrelation of Loss"):
    """
    Plots the averaged autocorrelation with a shaded region for standard deviation.

    Args:
        lags (List[float]): The list of lags.
        mean_autocorr (List[float]): The mean autocorrelation at each lag.
        std_autocorr (List[float]): The standard deviation of autocorrelation at each lag.
        title (str): The title of the plot.
    """
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

