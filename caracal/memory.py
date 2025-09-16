# caracal/memory.py - Production-ready memory management with backward compatibility

import gc
import os
import threading
import contextlib
import tempfile
import time
import traceback
from typing import Optional, Dict, Any, List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Core dependencies
import numpy as np
import pandas as pd

# Optional dependencies with graceful handling
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
    import torch

    HAS_PYTORCH = True
except ImportError:
    torch = None
    HAS_PYTORCH = False

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def _check_psutil():
    """Check psutil availability."""
    if not HAS_PSUTIL:
        raise ImportError("psutil required for memory monitoring. Install with: pip install psutil")


@dataclass
class MemorySnapshot:
    """Snapshot of memory state for comparison."""
    timestamp: float
    process_memory_mb: float
    gpu_memory_mb: Dict[str, float]
    system_available_gb: float
    tensorflow_memory_stats: Dict[str, Any]

    def memory_delta(self, other: 'MemorySnapshot') -> float:
        """Calculate memory difference in MB."""
        return self.process_memory_mb - other.process_memory_mb


class MemoryMonitor:
    """Monitors system memory usage and provides warnings/limits."""

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.initial_memory = self._get_memory_usage() if HAS_PSUTIL else {}

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not HAS_PSUTIL:
            return {}

        try:
            process = psutil.Process(os.getpid())

            return {
                'system_memory_percent': psutil.virtual_memory().percent / 100.0,
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'system_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
        except Exception:
            return {}

    def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status and return warnings if needed."""
        current = self._get_memory_usage()

        status = {
            'current_usage': current,
            'warnings': [],
            'critical': False
        }

        if not current:  # psutil not available
            return status

        if current['system_memory_percent'] > self.critical_threshold:
            status['critical'] = True
            status['warnings'].append(f"CRITICAL: System memory usage at {current['system_memory_percent']:.1%}")

        elif current['system_memory_percent'] > self.warning_threshold:
            status['warnings'].append(f"WARNING: High memory usage at {current['system_memory_percent']:.1%}")

        # Check for memory growth
        if self.initial_memory and 'process_memory_mb' in self.initial_memory:
            growth_mb = current['process_memory_mb'] - self.initial_memory['process_memory_mb']
            if growth_mb > 1000:  # More than 1GB growth
                status['warnings'].append(f"Process memory grew by {growth_mb:.0f}MB")

        return status


class BaseResourceCleaner(ABC):
    """Abstract base class for resource cleanup strategies."""

    @abstractmethod
    def cleanup(self) -> bool:
        """Perform cleanup. Returns True if successful."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage for this resource type."""
        pass


class TensorFlowCleaner(BaseResourceCleaner):
    """Comprehensive TensorFlow/Keras memory cleanup with aggressive GPU management."""

    def __init__(self):
        self.gpu_devices = []
        self._initialize_gpu_settings()

    def _initialize_gpu_settings(self):
        """Initialize GPU memory settings for optimal cleanup."""
        if not HAS_TENSORFLOW:
            return

        try:
            # Configure GPU memory growth
            self.gpu_devices = tf.config.list_physical_devices('GPU')

            if self.gpu_devices:
                for gpu in self.gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError:
                        # Already initialized - expected after first call
                        pass

        except Exception as e:
            print(f"GPU initialization warning: {e}")

    def cleanup(self) -> bool:
        """Perform comprehensive TensorFlow cleanup with aggressive GPU management."""
        if not HAS_TENSORFLOW:
            return True

        success_steps = []

        try:
            # Step 1: Clear Keras backend session
            tf.keras.backend.clear_session()
            success_steps.append("keras_session")

            # Step 2: Aggressive GPU cleanup
            self._cleanup_gpu_resources()
            success_steps.append("gpu_cleanup")

            return True

        except Exception as e:
            print(f"TensorFlow cleanup partially failed. Completed: {success_steps}. Error: {e}")
            return len(success_steps) > 0

    def _cleanup_gpu_resources(self):
        """More aggressive GPU cleanup with device reset."""
        if not HAS_TENSORFLOW or not self.gpu_devices:
            return

        try:
            # Clear all TensorFlow state
            tf.keras.backend.clear_session()

            # Force garbage collection
            import gc
            gc.collect()

            # Reset GPU devices (nuclear option)
            for gpu in self.gpu_devices:
                try:
                    # Reset all memory stats
                    tf.config.experimental.reset_memory_stats(gpu.name)

                    # Try to force memory deallocation
                    with tf.device(gpu.name):
                        # Create and immediately delete a small tensor to force cleanup
                        temp = tf.constant([[1.0]])
                        del temp

                except Exception as e:
                    print(f"GPU reset warning for {gpu.name}: {e}")

        except Exception:
            pass


    def get_memory_usage(self) -> Dict[str, Any]:
        """Get TensorFlow-specific memory usage including detailed GPU stats."""
        if not HAS_TENSORFLOW:
            return {}

        usage = {}

        try:
            for i, gpu in enumerate(self.gpu_devices):
                try:
                    memory_info = tf.config.experimental.get_memory_info(gpu.name)
                    usage[f'gpu_{i}_current_mb'] = memory_info['current'] / 1024 / 1024
                    usage[f'gpu_{i}_peak_mb'] = memory_info['peak'] / 1024 / 1024

                    # Check for fragmentation
                    current_mb = memory_info['current'] / 1024 / 1024
                    peak_mb = memory_info['peak'] / 1024 / 1024
                    usage[f'gpu_{i}_fragmentation_ratio'] = peak_mb / max(current_mb, 1)
                    usage[f'gpu_{i}_needs_cleanup'] = current_mb > 1000 or peak_mb / max(current_mb, 1) > 2

                except Exception:
                    usage[f'gpu_{i}_status'] = 'unavailable'

        except Exception:
            usage['gpu_info'] = 'unavailable'

        return usage


class PyTorchCleaner(BaseResourceCleaner):
    """PyTorch memory cleanup."""

    def cleanup(self) -> bool:
        if not HAS_PYTORCH:
            return True

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            return True
        except Exception:
            return False

    def get_memory_usage(self) -> Dict[str, Any]:
        if not HAS_PYTORCH:
            return {}

        usage = {}
        try:
            if torch.cuda.is_available():
                usage['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                usage['cuda_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        except Exception:
            pass

        return usage


class SystemCleaner(BaseResourceCleaner):
    """System-level memory cleanup."""

    def cleanup(self) -> bool:
        """Perform system-level cleanup."""
        try:
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()

            return True

        except Exception:
            return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get system memory usage."""
        if not HAS_PSUTIL:
            return {'gc_objects': len(gc.get_objects())}

        try:
            process = psutil.Process(os.getpid())

            return {
                'process_rss_mb': process.memory_info().rss / 1024 / 1024,
                'process_vms_mb': process.memory_info().vms / 1024 / 1024,
                'system_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'system_percent': psutil.virtual_memory().percent,
                'gc_objects': len(gc.get_objects())
            }
        except Exception:
            return {'gc_objects': len(gc.get_objects())}


class IsolationBackend(ABC):
    """Abstract base class for process isolation backends."""

    @abstractmethod
    def execute_isolated(self, func: Callable, data: Dict[str, Any],
                         timeout: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Execute function in isolated process/environment."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup backend resources."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of this backend."""
        pass


class JoblibBackend(IsolationBackend):
    """Joblib-based process isolation backend."""

    def __init__(self, n_jobs: int = 1, backend: str = 'multiprocessing'):
        if not HAS_JOBLIB:
            raise ImportError("joblib required for process isolation. Install with: pip install joblib")

        self.n_jobs = n_jobs
        self.backend = backend
        self._temp_dir = tempfile.mkdtemp(prefix='caracal_joblib_')

    @property
    def backend_name(self) -> str:
        return f"joblib({self.backend})"

    def execute_isolated(self, func: Callable, data: Dict[str, Any],
                         timeout: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Execute function using joblib process isolation."""

        @delayed
        def isolated_wrapper(func_data):
            """Wrapper that executes the function with proper error handling."""
            try:
                result = func(func_data)
                return {'success': True, 'result': result}
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }

        try:
            # Execute with joblib
            with Parallel(n_jobs=self.n_jobs, backend=self.backend, timeout=timeout) as parallel:
                results = parallel(isolated_wrapper(data))

            # Process results
            if results and len(results) > 0:
                result = results[0]
                return result['success'], result
            else:
                return False, {'error': 'No results returned from parallel execution'}

        except Exception as e:
            return False, {
                'error': f'Joblib execution failed: {str(e)}',
                'traceback': traceback.format_exc()
            }

    def cleanup(self):
        """Cleanup temporary directory."""
        try:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass


class MemoryManager:
    """Comprehensive memory management system for Caracal with backward compatibility."""

    def __init__(self,
                 enable_monitoring: bool = True,
                 cleanup_threshold: float = 0.85,
                 force_cleanup_threshold: float = 0.95,
                 enable_process_isolation: bool = False,
                 isolation_backend: str = 'joblib',
                 **isolation_kwargs):

        self.enable_monitoring = enable_monitoring and HAS_PSUTIL
        self.cleanup_threshold = cleanup_threshold
        self.force_cleanup_threshold = force_cleanup_threshold
        self.enable_process_isolation = enable_process_isolation

        # Initialize resource cleaners
        self.cleaners: List[BaseResourceCleaner] = [
            SystemCleaner(),
            TensorFlowCleaner(),
        ]

        if HAS_PYTORCH:
            self.cleaners.append(PyTorchCleaner())

        self.monitor = MemoryMonitor() if self.enable_monitoring else None

        # Process isolation manager (optional)
        self.process_manager = None
        if enable_process_isolation:
            try:
                self.process_manager = ProcessIsolationManager(isolation_backend, **isolation_kwargs)
            except ImportError as e:
                print(f"Warning: Could not initialize process isolation: {e}")
                self.enable_process_isolation = False

        # Memory tracking
        self.memory_snapshots = []
        self._lock = threading.Lock()

    def take_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a snapshot of current memory state."""
        snapshot_data = {
            'timestamp': time.time(),
            'process_memory_mb': 0,
            'gpu_memory_mb': {},
            'system_available_gb': 0,
            'tensorflow_memory_stats': {}
        }

        # System memory
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                snapshot_data['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
                snapshot_data['system_available_gb'] = psutil.virtual_memory().available / 1024 / 1024 / 1024
            except Exception:
                pass

        # GPU memory from TensorFlow cleaner
        for cleaner in self.cleaners:
            if isinstance(cleaner, TensorFlowCleaner):
                gpu_usage = cleaner.get_memory_usage()
                for key, value in gpu_usage.items():
                    if 'current_mb' in key:
                        snapshot_data['gpu_memory_mb'][key] = value

        snapshot = MemorySnapshot(**snapshot_data)

        with self._lock:
            self.memory_snapshots.append((label, snapshot))
            # Keep only last 50 snapshots
            if len(self.memory_snapshots) > 50:
                self.memory_snapshots = self.memory_snapshots[-50:]

        return snapshot

    def cleanup_all(self, force: bool = False) -> Dict[str, Any]:
        """Perform comprehensive cleanup of all resources - BACKWARD COMPATIBLE METHOD."""
        with self._lock:
            results = {
                'pre_cleanup_memory': self._get_total_memory_usage(),
                'cleanup_results': {},
                'post_cleanup_memory': {},
                'memory_freed_mb': 0
            }

            pre_memory = results['pre_cleanup_memory'].get('SystemCleaner_process_rss_mb', 0)

            # Run all cleaners
            for cleaner in self.cleaners:
                cleaner_name = cleaner.__class__.__name__
                try:
                    success = cleaner.cleanup()
                    results['cleanup_results'][cleaner_name] = {
                        'success': success,
                        'error': None
                    }
                except Exception as e:
                    results['cleanup_results'][cleaner_name] = {
                        'success': False,
                        'error': str(e)
                    }

            # Get post-cleanup memory
            results['post_cleanup_memory'] = self._get_total_memory_usage()
            post_memory = results['post_cleanup_memory'].get('SystemCleaner_process_rss_mb', 0)

            if pre_memory > 0 and post_memory > 0:
                results['memory_freed_mb'] = pre_memory - post_memory

            return results

    def _get_total_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage from all sources."""
        total_usage = {}

        for cleaner in self.cleaners:
            cleaner_name = cleaner.__class__.__name__
            try:
                usage = cleaner.get_memory_usage()
                for key, value in usage.items():
                    total_usage[f'{cleaner_name}_{key}'] = value
            except Exception as e:
                total_usage[f'{cleaner_name}_error'] = str(e)

        return total_usage

    def check_and_cleanup_if_needed(self) -> Optional[Dict[str, Any]]:
        """Check memory status and cleanup if thresholds are exceeded - BACKWARD COMPATIBLE."""
        if not self.monitor:
            return None

        status = self.monitor.check_memory_status()

        should_cleanup = (
                status['critical'] or
                status['current_usage'].get('system_memory_percent', 0) > self.cleanup_threshold
        )

        if should_cleanup:
            force = status['current_usage'].get('system_memory_percent', 0) > self.force_cleanup_threshold
            return self.cleanup_all(force=force)

        return None

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report - BACKWARD COMPATIBLE."""
        current_snapshot = self.take_memory_snapshot('current')

        report = {
            'detailed_usage': self._get_total_memory_usage(),
            'current_state': current_snapshot.__dict__,
            'snapshot_history_count': len(self.memory_snapshots),
            'process_isolation_available': self.enable_process_isolation
        }

        if HAS_PSUTIL:
            try:
                report['system_info'] = {
                    'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    'cpu_count': psutil.cpu_count()
                }
            except Exception:
                pass

        if self.monitor:
            report['status'] = self.monitor.check_memory_status()

        return report


class ProcessIsolationManager:
    """Manages process isolation with pluggable backends."""

    def __init__(self, backend: str = 'joblib', **backend_kwargs):
        self.backend_name = backend
        self.backend = self._create_backend(backend, **backend_kwargs)

    def _create_backend(self, backend: str, **kwargs) -> IsolationBackend:
        """Factory method to create appropriate backend."""
        if backend == 'joblib':
            return JoblibBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Available: 'joblib'")

    def run_isolated_training(self, training_function: Callable,
                              training_data: Dict[str, Any],
                              timeout: Optional[int] = 3600) -> Tuple[bool, Dict[str, Any]]:
        """Run training function in isolated process."""
        return self.backend.execute_isolated(training_function, training_data, timeout)

    def cleanup(self):
        """Cleanup backend resources."""
        if self.backend:
            self.backend.cleanup()

    def __del__(self):
        """Ensure cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


@contextlib.contextmanager
def managed_memory_context(auto_cleanup: bool = True,
                           cleanup_threshold: float = 0.8):
    """Context manager for automatic memory management - BACKWARD COMPATIBLE NAME."""

    manager = MemoryManager(
        enable_monitoring=HAS_PSUTIL,
        cleanup_threshold=cleanup_threshold
    )

    try:
        yield manager

    finally:
        if auto_cleanup:
            cleanup_results = manager.cleanup_all()

            # Log significant memory issues
            memory_freed = cleanup_results.get('memory_freed_mb', 0)
            if memory_freed > 500:  # More than 500MB freed
                print(f"Memory cleanup freed {memory_freed:.0f}MB")

            cleanup_results_dict = cleanup_results.get('cleanup_results', {})
            if any(not result.get('success', True) for result in cleanup_results_dict.values()):
                failed_cleaners = [name for name, result in cleanup_results_dict.items()
                                   if not result.get('success', True)]
                print(f"Warning: Some cleaners failed: {failed_cleaners}")