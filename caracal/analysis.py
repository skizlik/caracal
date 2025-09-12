# caracal/analysis.py


import gc
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import beta, f_oneway, kruskal, mannwhitneyu, shapiro, wilcoxon

# Local application imports
from .config import ModelConfig
from .core import BaseModelWrapper
from .loggers import BaseLogger
from .runners import ExperimentRunner

def get_confusion_matrix_df(predictions: np.ndarray, true_labels: np.ndarray,
                            class_names: Dict[int, str]) -> pd.DataFrame:
    """Generate confusion matrix DataFrame from predictions and true labels."""
    from sklearn.metrics import confusion_matrix

    class_indices = sorted(class_names.keys())
    cm = confusion_matrix(true_labels, predictions, labels=class_indices)

    return pd.DataFrame(
        cm,
        index=[class_names[i] for i in class_indices],
        columns=[class_names[i] for i in class_indices]
    )

# CONVERGENCE ASSESSMENT

def calculate_autocorr(data: Union[pd.Series, List[float]], lag: int = 1) -> Optional[float]:
    """
    Calculates the autocorrelation of a time series at a specific lag.

    Args:
        data (pd.Series or List[float]): The time series data (e.g., loss history).
        lag (int): The number of time steps to lag.

    Returns:
        Optional[float]: The autocorrelation value, or None if the data is too short.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) <= lag:
        return None

    return data.autocorr(lag)


def calculate_averaged_autocorr(histories: List[pd.Series], max_lag: int = 20) -> Tuple[
    List[float], List[float], List[float]]:
    """
    Computes the mean and standard deviation of the autocorrelation functions
    of multiple training histories.

    Args:
        histories (List[pd.Series]): A list of metric histories from each run.
        max_lag (int): The maximum lag to compute autocorrelation for.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing
        (lags, mean_autocorr, std_autocorr).
    """
    if not histories:
        raise ValueError("The list of histories cannot be empty.")

    autocorrs = []
    for history in histories:
        autocorr_values = [history.autocorr(lag) for lag in range(1, max_lag + 1)]
        autocorrs.append(autocorr_values)

    autocorrs_np = np.array(autocorrs)
    mean_autocorr = np.nanmean(autocorrs_np, axis=0)
    std_autocorr = np.nanstd(autocorrs_np, axis=0)

    lags = list(range(1, max_lag + 1))

    return lags, mean_autocorr, std_autocorr


def check_convergence(data: Union[pd.Series, List[float]], window_size: int = 5,
                      autocorr_threshold: float = 0.1) -> bool:
    """
    Checks for convergence by assessing the autocorrelation of the data.

    Args:
        data (pd.Series or List[float]): The time series data (e.g., loss history).
        window_size (int): The number of recent epochs to check for convergence.
        autocorr_threshold (float): The threshold below which autocorrelation is considered
                                    a sign of convergence.

    Returns:
        bool: True if the model has converged, False otherwise.
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) < window_size + 1:
        return False

    # Get the data from the most recent window of epochs
    recent_data = data.iloc[-window_size:]

    # Calculate autocorrelation at a lag of 1
    autocorr = calculate_autocorr(recent_data, lag=1)

    # Check if the autocorrelation is below the threshold
    return autocorr is not None and autocorr < autocorr_threshold


# HYPOTHESIS TESTS

def mann_whitney_test(model1_metrics: pd.Series, model2_metrics: pd.Series, alternative: str = 'two-sided') -> Dict[
    str, Any]:
    """
    Performs a Mann-Whitney U test to compare the performance of two models.

    Args:
        model1_metrics (pd.Series): A series of performance metrics from the first model.
        model2_metrics (pd.Series): A series of performance metrics from the second model.
        alternative (str): The alternative hypothesis. Options are 'two-sided', 'less', or 'greater'.

    Returns:
        Dict[str, Any]: A dictionary containing the U-statistic, p-value, and a clear
                        conclusion.
    """
    if not isinstance(model1_metrics, pd.Series) or not isinstance(model2_metrics, pd.Series):
        raise TypeError("Inputs must be pandas Series.")

    # Perform the Mann-Whitney U test
    statistic, pvalue = mannwhitneyu(model1_metrics, model2_metrics, alternative=alternative)

    # Formulate a clear conclusion
    conclusion = (
        f"The Mann-Whitney U test suggests there is a statistically significant "
        f"difference in model performance (p={pvalue:.4f}) at the 0.05 level."
        if pvalue < 0.05
        else (
            f"The Mann-Whitney U test suggests there is no statistically significant "
            f"difference in model performance (p={pvalue:.4f}) at the 0.05 level."
        )
    )

    # Return the results in a dictionary
    return {
        'statistic': statistic,
        'p-value': pvalue,
        'alternative': alternative,
        'conclusion': conclusion
    }



def wilcoxon_signed_rank_test(model_metrics: pd.Series, null_value: float = 0.5, alternative: str = 'two-sided') -> \
Dict[str, Any]:
    """
    Performs a one-sample Wilcoxon signed-rank test on model performance metrics.

    Args:
        model_metrics (pd.Series): A series of performance metrics from a single model.
        null_value (float): The value to test the metrics against (the null hypothesis).
        alternative (str): The alternative hypothesis. Options are 'two-sided', 'less', or 'greater'.

    Returns:
        Dict[str, Any]: A dictionary containing the test statistic, p-value, and conclusion.
    """
    # The Wilcoxon test compares samples to a median. We subtract the null value.
    if not isinstance(model_metrics, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    # Perform the Wilcoxon signed-rank test
    statistic, pvalue = wilcoxon(model_metrics - null_value, alternative=alternative)

    # Formulate a clear conclusion
    conclusion = (
        f"The Wilcoxon test suggests the model's performance is statistically "
        f"significantly different from {null_value} (p={pvalue:.4f}) at the 0.05 level."
        if pvalue < 0.05
        else (
            f"The Wilcoxon test suggests no statistically significant difference from "
            f"{null_value} (p={pvalue:.4f}) at the 0.05 level."
        )
    )

    # Return the results in a dictionary
    return {
        'statistic': statistic,
        'p-value': pvalue,
        'alternative': alternative,
        'conclusion': conclusion
    }


def anova_test(model_metrics: Dict[str, pd.Series], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Performs a one-way ANOVA test to compare the means of multiple models.

    Args:
        model_metrics (Dict[str, pd.Series]): A dictionary where keys are model names and
                                              values are series of performance metrics.
        alpha (float): The significance level.

    Returns:
        Dict[str, Any]: A dictionary with the F-statistic, p-value, and a conclusion.
    """
    if len(model_metrics) < 2:
        raise ValueError("ANOVA requires at least two groups to compare.")

    # Convert the dictionary values to a list for the test
    metric_series_list = list(model_metrics.values())

    statistic, pvalue = f_oneway(*metric_series_list)

    conclusion = (
        f"The ANOVA test suggests a statistically significant difference in model means "
        f"(p={pvalue:.4f}) at the {alpha} level."
        if pvalue < alpha
        else (
            f"The ANOVA test suggests no statistically significant difference in model means "
            f"(p={pvalue:.4f}) at the {alpha} level."
        )
    )

    return {
        'statistic': statistic,
        'p-value': pvalue,
        'conclusion': conclusion
    }



def kruskal_wallis_test(model_metrics: Dict[str, pd.Series], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Performs a Kruskal-Wallis H-test to compare the medians of multiple models.

    Args:
        model_metrics (Dict[str, pd.Series]): A dictionary where keys are model names and
                                              values are series of performance metrics.
        alpha (float): The significance level.

    Returns:
        Dict[str, Any]: A dictionary with the H-statistic, p-value, and a conclusion.
    """
    if len(model_metrics) < 2:
        raise ValueError("Kruskal-Wallis test requires at least two groups to compare.")

    # Convert the dictionary values to a list for the test
    metric_series_list = list(model_metrics.values())

    statistic, pvalue = kruskal(*metric_series_list)

    conclusion = (
        f"The Kruskal-Wallis test suggests a statistically significant difference in model medians "
        f"(p={pvalue:.4f}) at the {alpha} level."
        if pvalue < alpha
        else (
            f"The Kruskal-Wallis test suggests no statistically significant difference in model medians "
            f"(p={pvalue:.4f}) at the {alpha} level."
        )
    )

    return {
        'statistic': statistic,
        'p-value': pvalue,
        'conclusion': conclusion
    }


def shapiro_wilk_test(model_metrics: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Performs the Shapiro-Wilk test to check if a sample comes from a normal distribution.

    Args:
        model_metrics (pd.Series): A series of performance metrics from a single model.
        alpha (float): The significance level.

    Returns:
        Dict[str, Any]: A dictionary containing the test statistic, p-value, and conclusion.
    """
    if not isinstance(model_metrics, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    statistic, pvalue = shapiro(model_metrics)

    conclusion = (
        f"The Shapiro-Wilk test suggests the sample is not normally distributed "
        f"(p={pvalue:.4f}) at the {alpha} level."
        if pvalue < alpha
        else (
            f"The Shapiro-Wilk test suggests the sample is likely normally distributed "
            f"(p={pvalue:.4f}) at the {alpha} level."
        )
    )

    return {
        'statistic': statistic,
        'p-value': pvalue,
        'conclusion': conclusion
    }