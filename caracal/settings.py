# caracal/settings.py
import logging
import sys
from typing import Union

# Global Configuration
_DISPLAY_PLOTS: bool = True
_VERBOSE: bool = True

# Setup Library Logger
logger = logging.getLogger("caracal")
logger.setLevel(logging.INFO)

# Default handler (StreamHandler to stdout)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('%(message)s'))  # Simple format like print()
logger.addHandler(_handler)

def set_verbose(verbose: bool):
    """
    Global override for verbosity.
    
    Args:
        verbose: If True, log level is INFO. If False, WARNING.
    """
    global _VERBOSE
    _VERBOSE = verbose
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)

def set_display_plots(display: bool):
    """
    Global override for plot display.
    
    Args:
        display: If True, plotting functions will call plt.show().
                 If False, they will return the Figure without blocking.
    """
    global _DISPLAY_PLOTS
    _DISPLAY_PLOTS = display

def should_display() -> bool:
    """Internal check for plotting functions."""
    return _DISPLAY_PLOTS
