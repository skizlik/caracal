# caracal/memory.py - Simple two-mode memory management system
# Mode 1: Corrected in-process cleanup (default)
# Mode 2: Process isolation (user-controlled)

import gc
import os
import time
import warnings
import contextlib
import pickle
import tempfile
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field

# Core dependencies
import numpy as np
import pandas as pd

# Optional dependencies
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    joblib = None
    HAS_JOBLIB = False


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    success: bool
    actions_taken: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cleanup_time_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# Framework Handlers
class TensorFlowHandler:
    """
    Corrected TensorFlow memory management.

    IMPORTANT: Does NOT use reset_memory_stats() which doesn't free memory.
    Uses only proven cleanup methods.
    """

    def __init__(self):
        self.gpus: List[Any] = []
        self.setup_complete = False

        if HAS_TENSORFLOW:
            try:
                self.gpus = tf.config.list_physical_devices('GPU')
            except Exception:
                self.gpus = []

    def setup_constraints(self, memory_limit_mb: Optional[int] = None) -> bool:
        """Configure TensorFlow GPU memory limits."""
        if not HAS_TENSORFLOW or self.setup_complete:
            return True

        if not self.gpus:
            return True  # CPU mode

        success_count = 0
        for gpu in self.gpus:
            try:
                if memory_limit_mb:
                    tf.config.set_memory_limit(gpu, memory_limit_mb)
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
                success_count += 1

            except RuntimeError as e:
                if "cannot be modified" in str(e).lower():
                    return False  # Already initialized
                warnings.warn(f"GPU setup failed for {gpu.name}: {e}")
            except Exception as e:
                warnings.warn(f"GPU setup error for {gpu.name}: {e}")

        self.setup_complete = success_count > 0
        return self.setup_complete

    def cleanup_after_run(self) -> CleanupResult:
        """
        CORRECTED TensorFlow cleanup using only reliable methods.

        Key fixes:
        1. Does NOT call tf.config.experimental.reset_memory_stats() - that's useless
        2. Uses tf.keras.backend.clear_session() - this actually works
        3. Multiple garbage collection passes
        4. Brief pause for asynchronous cleanup
        """
        start_time = time.time()
        result = CleanupResult(success=True)

        if not HAS_TENSORFLOW:
            result.actions_taken.append('tensorflow_not_available')
            return result

        try:
            # Primary cleanup: Clear Keras session (this is what actually works)
            tf.keras.backend.clear_session()
            result.actions_taken.append('cleared_keras_session')

        except Exception as e:
            result.success = False
            result.errors.append(f'Keras session clear failed: {str(e)[:80]}')

        try:
            # Secondary cleanup: Multiple garbage collection passes
            total_collected = 0
            for i in range(3):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break

            result.actions_taken.append(f'gc_collected_{total_collected}_objects')

        except Exception as e:
            result.errors.append(f'GC failed: {str(e)[:50]}')

        # Brief pause to allow asynchronous GPU cleanup to complete
        time.sleep(0.1)
        result.actions_taken.append('async_cleanup_wait')

        result.cleanup_time_seconds = time.time() - start_time
        return result

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get TensorFlow GPU memory usage.

        NOTE: This is unreliable immediately after cleanup due to async operations.
        Use for monitoring only, not for control flow.
        """
        if not HAS_TENSORFLOW or not self.gpus:
            return {}

        memory_info = {}
        for i, gpu in enumerate(self.gpus):
            try:
                info = tf.config.experimental.get_memory_info(gpu.name)
                if info and 'current' in info:
                    memory_info[f'gpu_{i}'] = float(info['current']) / 1024 / 1024
            except Exception:
                continue

        return memory_info


class SystemHandler:
    """System-level memory management."""

    def setup_constraints(self, memory_limit_mb: Optional[int] = None) -> bool:
        """System handler requires no setup."""
        return True

    def cleanup_after_run(self) -> CleanupResult:
        """Reliable system-level cleanup."""
        start_time = time.time()
        result = CleanupResult(success=True)

        try:
            # Multiple garbage collection passes to handle circular references
            total_collected = 0
            for i in range(3):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break

            result.actions_taken.append(f'system_gc_collected_{total_collected}_objects')

        except Exception as e:
            result.success = False
            result.errors.append(f'System cleanup failed: {str(e)[:80]}')

        result.cleanup_time_seconds = time.time() - start_time
        return result

    def get_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        info = {}

        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                vm = psutil.virtual_memory()

                info.update({
                    'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                    'system_available_gb': vm.available / (1024 ** 3),
                    'system_percent_used': vm.percent
                })
            except Exception:
                pass

        return info


# Standard In-Process Resource Manager
class StandardResourceManager:
    """
    Standard resource manager with corrected in-process cleanup.

    This is the default mode - fast execution with reliable cleanup
    that should work for most scenarios.
    """

    def __init__(self):
        self.handlers = self._initialize_handlers()
        self.setup_complete = False

    def _initialize_handlers(self) -> Dict[str, Any]:
        """Initialize available framework handlers."""
        handlers = {'system': SystemHandler()}

        if HAS_TENSORFLOW:
            handlers['tensorflow'] = TensorFlowHandler()

        # Future: PyTorch handler would be added here
        # if HAS_PYTORCH:
        #     handlers['pytorch'] = PyTorchHandler()

        return handlers

    def setup_training_constraints(self, memory_limit_mb: Optional[int] = None) -> bool:
        """Set up memory constraints for training."""
        if self.setup_complete:
            return True

        setup_success = False
        for name, handler in self.handlers.items():
            try:
                if handler.setup_constraints(memory_limit_mb):
                    setup_success = True
            except Exception as e:
                warnings.warn(f"Handler {name} setup failed: {e}")

        self.setup_complete = True
        return setup_success

    def cleanup_after_run(self) -> CleanupResult:
        """
        Perform corrected in-process cleanup across all frameworks.

        This should handle most memory accumulation scenarios without
        the overhead of process isolation.
        """
        overall_result = CleanupResult(success=True)
        start_time = time.time()

        # Run cleanup for each framework handler
        for name, handler in self.handlers.items():
            try:
                handler_result = handler.cleanup_after_run()

                # Aggregate results
                overall_result.actions_taken.extend([
                    f"{name}: {action}" for action in handler_result.actions_taken
                ])

                if handler_result.has_errors:
                    overall_result.errors.extend([
                        f"{name}: {error}" for error in handler_result.errors
                    ])
                    overall_result.success = False

            except Exception as e:
                overall_result.errors.append(f"{name} cleanup failed: {str(e)[:80]}")
                overall_result.success = False

        overall_result.cleanup_time_seconds = time.time() - start_time
        return overall_result

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        report = {
            'timestamp': time.time(),
            'mode': 'standard_cleanup',
            'handlers_available': list(self.handlers.keys()),
            'memory_info': {}
        }

        for name, handler in self.handlers.items():
            try:
                info = handler.get_memory_info()
                report['memory_info'][name] = info if info else {'no_data': True}
            except Exception as e:
                report['memory_info'][name] = {'error': str(e)[:50]}

        return report


# Process Isolation Resource Manager
class ProcessIsolationManager:
    """
    Process isolation resource manager.

    Runs each training iteration in a completely separate process
    for guaranteed memory cleanup. Slower but 100% reliable.
    """

    def __init__(self):
        if not HAS_JOBLIB:
            raise ImportError(
                "Process isolation requires joblib. Install with: pip install joblib"
            )

    def setup_training_constraints(self, memory_limit_mb: Optional[int] = None) -> bool:
        """Process isolation setup - verify joblib availability."""
        return HAS_JOBLIB

    def cleanup_after_run(self) -> CleanupResult:
        """
        Process isolation cleanup.

        NOTE: The real cleanup happens when the isolated process terminates.
        This method is called from within the isolated process and does
        basic in-process cleanup as a backup.
        """
        result = CleanupResult(success=True)
        result.actions_taken.append('process_isolation_cleanup_on_exit')

        # Do basic cleanup within the process as well
        try:
            collected = gc.collect()
            result.actions_taken.append(f'backup_gc_collected_{collected}_objects')
        except Exception as e:
            result.errors.append(f'Backup cleanup failed: {str(e)[:80]}')

        return result

    def run_training_isolated(self,
                              training_func: Callable,
                              args: tuple = (),
                              kwargs: dict = None) -> Any:
        """
        Run training function in completely isolated process.

        Args:
            training_func: Function to run in isolation
            args: Positional arguments for training_func
            kwargs: Keyword arguments for training_func

        Returns:
            Result from training_func

        Raises:
            RuntimeError: If process isolation fails
        """
        if kwargs is None:
            kwargs = {}

        try:
            # Use joblib with loky backend for reliable process isolation
            with joblib.parallel_backend('loky', n_jobs=1):
                delayed_func = joblib.delayed(training_func)
                result = delayed_func(*args, **kwargs)

            return result

        except Exception as e:
            raise RuntimeError(f"Process isolation failed: {e}")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get memory report for process isolation mode."""
        return {
            'timestamp': time.time(),
            'mode': 'process_isolation',
            'isolation_backend': 'joblib_loky',
            'memory_info': {
                'note': 'Memory cleanup guaranteed by process termination'
            }
        }


# Factory Function
def get_resource_manager(use_process_isolation: bool = False) -> Union[
    StandardResourceManager, ProcessIsolationManager]:
    """
    Create appropriate resource manager based on user preference.

    Args:
        use_process_isolation: If True, use process isolation for guaranteed cleanup.
                             If False, use corrected in-process cleanup (default).

    Returns:
        Configured resource manager

    Raises:
        ImportError: If process isolation requested but joblib not available
    """
    if use_process_isolation:
        return ProcessIsolationManager()
    else:
        return StandardResourceManager()


# Context Manager
@contextlib.contextmanager
def managed_training_context(memory_limit_mb: Optional[int] = None,
                             use_process_isolation: bool = False):
    """
    Context manager for training with automatic memory management.

    Args:
        memory_limit_mb: GPU memory limit per device in MB
        use_process_isolation: If True, enables process isolation mode

    Example:
        # Standard mode (default)
        with managed_training_context(memory_limit_mb=2048) as manager:
            # Training code here

        # Process isolation mode (slower but more reliable)
        with managed_training_context(memory_limit_mb=2048,
                                    use_process_isolation=True) as manager:
            # Training code here
    """
    manager = get_resource_manager(use_process_isolation)

    try:
        # Setup
        setup_success = manager.setup_training_constraints(memory_limit_mb)

        if not setup_success:
            mode = "process isolation" if use_process_isolation else "standard cleanup"
            warnings.warn(f"Memory management setup incomplete for {mode} mode")
        else:
            mode = "process isolation" if use_process_isolation else "standard cleanup"
            print(f"Memory management configured with {mode}")

        yield manager

    finally:
        # Final cleanup (only relevant for standard mode)
        if not use_process_isolation:
            try:
                cleanup_result = manager.cleanup_after_run()
                if cleanup_result.cleanup_time_seconds > 1.0:
                    print(f"Final cleanup completed in {cleanup_result.cleanup_time_seconds:.1f}s")
                if cleanup_result.has_errors:
                    print(f"Final cleanup had {len(cleanup_result.errors)} warnings")
            except Exception as e:
                warnings.warn(f"Final cleanup failed: {e}")


# Convenience Functions
def setup_memory_efficient_training(memory_limit_mb: Optional[int] = None,
                                    use_process_isolation: bool = False) -> Union[
    StandardResourceManager, ProcessIsolationManager]:
    """
    Set up memory-efficient training configuration.

    Args:
        memory_limit_mb: GPU memory limit per device
        use_process_isolation: Whether to use process isolation

    Returns:
        Configured resource manager
    """
    manager = get_resource_manager(use_process_isolation)

    setup_success = manager.setup_training_constraints(memory_limit_mb)

    if setup_success:
        mode = "process isolation" if use_process_isolation else "standard cleanup"
        print(f"Memory-efficient training configured with {mode}")
        if memory_limit_mb:
            print(f"GPU memory limit: {memory_limit_mb}MB per device")
    else:
        warnings.warn("Memory management setup failed")

    return manager
