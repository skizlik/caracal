# caracal/loggers.py

import mlflow
from typing import Dict, Any, List, Optional

class BaseLogger:
    """
    A base class for logging experiment data.
    
    The logger stores all logged data in a history object and can optionally
    print it to the console.
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history: Dict[str, Any] = {'params': {}, 'metrics': []}

    def log_params(self, params: Dict[str, Any]):
        """Logs a dictionary of parameters and stores it in history."""
        self.history['params'].update(params)
        if self.verbose:
            print(f"Parameters: {self.history['params']}")

    def log_metric(self, key: str, value: float, step: int = 0):
        """Logs a single metric and stores it in history."""
        metric_entry = {'key': key, 'value': value, 'step': step}
        self.history['metrics'].append(metric_entry)
        if self.verbose:
            print(f" - Step {step}: {key} = {value:.4f}")

    def end_run(self):
        """Called at the end of a run to finalize logging."""
        if self.verbose:
            print("-" * 50)
            
    def get_history(self) -> Dict[str, Any]:
        """Returns the logged history."""
        return self.history

class MLflowLogger(BaseLogger):
    """An implementation of the BaseLogger for MLflow."""
    def __init__(self, run_name: str, verbose: bool = True):
        super().__init__(verbose)
        self.run_name = run_name
        mlflow.start_run(run_name=self.run_name)

    def log_params(self, params: Dict[str, Any]):
        super().log_params(params)
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int = 0):
        super().log_metric(key, value, step=step)
        mlflow.log_metric(key, value, step=step)

    def end_run(self):
        super().end_run()
        mlflow.end_run()
