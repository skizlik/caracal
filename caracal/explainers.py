# caracal/explainers.py

import shap
import numpy as np
from typing import Optional, List, Any
import matplotlib.pyplot as plt

from .core import BaseModelWrapper

def plot_shap_summary(model_wrapper: BaseModelWrapper, X_data: np.ndarray, feature_names: Optional[List[str]] = None,
                      plot_type: str = "bar"):
    """
    Generates a SHAP summary plot for a model.
    
    Args:
        model_wrapper (BaseModelWrapper): The trained model wrapper.
        X_data (np.ndarray): The data to explain.
        feature_names (Optional[List[str]]): List of feature names.
        plot_type (str): The type of plot to generate ("bar", "dot", "violin").
    """
    model_type = type(model_wrapper.model).__name__

    if "Tree" in model_type:
        explainer = shap.TreeExplainer(model_wrapper.model)
        shap_values = explainer.shap_values(X_data)
        
    # elif "NeuralNet" in model_type:
        # For neural networks, you would need to use shap.DeepExplainer
        # This requires a TensorFlow backend, so it would be in the Keras-specific wrapper
        # explainer = shap.DeepExplainer(model_wrapper.model, X_data_background)
        # shap_values = explainer.shap_values(X_data)
        
    else:
        print("Using KernelExplainer, which can be very slow for large datasets.")
        explainer = shap.KernelExplainer(model_wrapper.model.predict_proba, X_data[:50])
        shap_values = explainer.shap_values(X_data)

    if isinstance(shap_values, list):
        # For multi-class models, SHAP returns a list of arrays.
        # We can't plot this directly, so we'll just plot the first class for now.
        shap.summary_plot(shap_values[0], X_data, feature_names=feature_names, plot_type=plot_type)
    else:
        shap.summary_plot(shap_values, X_data, feature_names=feature_names, plot_type=plot_type)
