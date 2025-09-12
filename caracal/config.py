# caracal/config.py

from typing import Dict, Any, Optional

class ModelConfig:
    """
    A class to manage and store all model training parameters and hyperparameters.
    
    This object is designed to be flexible and can store any number of key-value pairs.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the config with a dictionary of parameters."""
        self.params = params if params is not None else {}

    def __repr__(self) -> str:
        """Provides a clean string representation."""
        return f"ModelConfig({self.params})"
        
    def __getattr__(self, name):
        """Allows direct access to parameters like `config.epochs`."""
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def __getitem__(self, key):
        """Allows dictionary-style access like `config['epochs']`."""
        return self.params[key]

    def merge(self, other_params: Dict[str, Any]):
        """Merges a dictionary of new parameters into the config."""
        self.params.update(other_params)
        
    # --- Factory Methods for Smart Defaults ---
    
    @classmethod
    def from_defaults(cls) -> 'ModelConfig':
        """Returns a config with general, reasonable defaults."""
        return cls({
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'verbose': 0
        })

    @classmethod
    def for_cnn(cls, input_shape: tuple = (128, 128, 3), num_classes: int = 10) -> 'ModelConfig':
        """Returns a config with defaults suitable for a CNN model."""
        base_config = cls.from_defaults()
        base_config.merge({
            'input_shape': input_shape,
            'num_classes': num_classes,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        })
        return base_config

    @classmethod
    def for_xgboost(cls, num_classes: int = 10) -> 'ModelConfig':
        """Returns a config with defaults suitable for an XGBoost model."""
        base_config = cls.from_defaults()
        base_config.merge({
            'n_estimators': 180,
            'max_depth': 7,
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        })
        return base_config
