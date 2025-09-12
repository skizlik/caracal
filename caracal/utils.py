# caracal/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Union, Tuple, List, Any

def save_object(obj: Any, path: str):
    """Saves a Python object to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {path}")

def load_object(path: str) -> Any:
    """Loads a Python object from a file using pickle."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {path}")
    return obj

def train_val_test_split(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                         test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Splits data into training, validation, and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
