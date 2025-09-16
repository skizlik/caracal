# caracal/memory.py - GPU-focused memory management for deep learning workloads

import gc
import os
import contextlib
import time
from typing import Optional, Dict, Any, List

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


def setup_tensorflow_gpu(memory_limit_mb: Optional[int] = None) -> bool:
    """
    Configure TensorFlow GPU for reliable repeated training runs.

    Sets memory growth and optional memory limits to prevent OOM errors
    during variability studies with large models.

    Args:
        memory_limit_mb: Maximum GPU memory to use in MB. If None, uses growth mode.

    Returns:
        True if GPU configuration succeeded, False otherwise.
    """
    if not HAS_TENSORFLOW:
        return False

    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return True  # CPU mode is fine

        for gpu in gpus:
            if memory_limit_mb:
                tf.config.set_memory_limit(gpu, memory_limit_mb)
            else:
                tf.config.experimental.set_memory_growth(gpu, True)

        return True

    except RuntimeError:
        # GPU already initialized - can't change settings
        return False
    except Exception:
        return False


def set_tensorflow_env_vars():
    """
    Set environment variables to prevent TensorFlow GPU memory issues.

    Must be called before importing TensorFlow. Sets flags to:
    - Enable memory growth
    - Use async GPU allocator for better fragmentation handling
    - Use fallback convolution algorithms
    - Reduce log verbosity
    """
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def deep_clean_gpu():
    """
    Aggressive GPU memory cleanup for repeated training runs.

    Performs comprehensive cleanup including:
    - TensorFlow session clearing
    - GPU memory reset attempts
    - Garbage collection
    - Cache clearing

    Returns:
        Dict with cleanup results and memory freed.
    """
    results = {'actions': [], 'memory_freed_mb': 0}

    if not HAS_TENSORFLOW:
        results['actions'].append('TensorFlow not available')
        return results

    # Get initial GPU memory
    initial_memory = _get_gpu_memory_usage()

    try:
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        results['actions'].append('Cleared TensorFlow session')

        # Clear any cached operations
        if hasattr(tf.config.experimental, 'reset_memory_stats'):
            gpus = tf.config.list_physical_devices('GPU')
            for i, gpu in enumerate(gpus):
                try:
                    # Use correct device format for TensorFlow
                    device_name = f'GPU:{i}'
                    tf.config.experimental.reset_memory_stats(device_name)
                    results['actions'].append(f'Reset memory stats for {device_name}')
                except Exception as e:
                    # Don't fail the whole cleanup if one GPU fails
                    results['actions'].append(f'GPU {i} reset failed: {str(e)[:50]}')

        # Aggressive garbage collection
        for _ in range(3):
            collected = gc.collect()
            if collected > 0:
                results['actions'].append(f'GC collected {collected} objects')

        # Calculate memory freed
        final_memory = _get_gpu_memory_usage()
        if initial_memory and final_memory:
            for gpu_id in initial_memory:
                if gpu_id in final_memory:
                    freed = initial_memory[gpu_id] - final_memory[gpu_id]
                    if freed > 0:
                        results['memory_freed_mb'] += freed

    except Exception as e:
        results['actions'].append(f'Deep clean error: {str(e)}')

    return results


def _get_gpu_memory_usage() -> Dict[int, float]:
    """Get current GPU memory usage in MB for each GPU."""
    if not HAS_TENSORFLOW:
        return {}

    gpu_memory = {}
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(gpus):
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                gpu_memory[i] = memory_info['current'] / 1024 / 1024
            except Exception:
                pass
    except Exception:
        pass

    return gpu_memory


def _get_process_memory() -> float:
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0

    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


class MemoryManager:
    """
    Memory management for ML training workloads.

    Provides both lightweight cleanup for normal use and aggressive
    deep cleaning for GPU-heavy repeated training scenarios.
    """

    def __init__(self, enable_monitoring: bool = True, cleanup_threshold: float = 0.85,
                 force_cleanup_threshold: float = 0.95, **kwargs):
        """
        Initialize memory manager.

        Args:
            enable_monitoring: Whether to monitor system memory usage
            cleanup_threshold: Memory usage fraction that triggers cleanup
            force_cleanup_threshold: Memory usage fraction that triggers deep clean
        """
        self.enable_monitoring = enable_monitoring and HAS_PSUTIL
        self.cleanup_threshold = cleanup_threshold
        self.force_cleanup_threshold = force_cleanup_threshold

    def cleanup_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup with optional deep cleaning.

        Args:
            force: If True, performs aggressive GPU cleanup

        Returns:
            Cleanup results including memory freed and actions taken
        """
        pre_memory = _get_process_memory()

        results = {
            'pre_cleanup_memory': {'process_memory_mb': pre_memory},
            'cleanup_results': {},
            'post_cleanup_memory': {},
            'memory_freed_mb': 0
        }

        try:
            if force:
                # Aggressive cleanup for GPU workloads
                deep_results = deep_clean_gpu()
                results['cleanup_results']['deep_clean'] = {
                    'success': True,
                    'actions': deep_results['actions'],
                    'gpu_memory_freed_mb': deep_results['memory_freed_mb']
                }
            else:
                # Standard cleanup
                collected = gc.collect()
                if HAS_TENSORFLOW:
                    tf.keras.backend.clear_session()

                results['cleanup_results']['standard_clean'] = {
                    'success': True,
                    'gc_collected': collected
                }

            # Calculate total memory change
            post_memory = _get_process_memory()
            results['post_cleanup_memory'] = {'process_memory_mb': post_memory}

            if pre_memory > 0 and post_memory > 0:
                results['memory_freed_mb'] = max(0, pre_memory - post_memory)

        except Exception as e:
            results['cleanup_results']['error'] = {'success': False, 'error': str(e)}

        return results

    def check_and_cleanup_if_needed(self) -> Optional[Dict[str, Any]]:
        """
        Check memory usage and perform cleanup if thresholds exceeded.

        Returns:
            Cleanup results if cleanup was performed, None otherwise
        """
        if not self.enable_monitoring:
            return None

        try:
            memory_percent = psutil.virtual_memory().percent / 100.0

            if memory_percent > self.force_cleanup_threshold:
                return self.cleanup_all(force=True)
            elif memory_percent > self.cleanup_threshold:
                return self.cleanup_all(force=False)

        except Exception:
            pass

        return None

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage report.

        Returns:
            Memory usage statistics for system, process, and GPUs
        """
        report = {
            'process_memory_mb': _get_process_memory(),
            'gpu_memory': _get_gpu_memory_usage(),
            'cleanup_available': True
        }

        if HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                report['system_memory'] = {
                    'total_gb': vm.total / 1024 / 1024 / 1024,
                    'available_gb': vm.available / 1024 / 1024 / 1024,
                    'percent_used': vm.percent
                }
            except Exception:
                pass

        return report

    def take_memory_snapshot(self, label: str = "") -> Dict[str, Any]:
        """
        Capture current memory state for comparison.

        Args:
            label: Optional label for the snapshot

        Returns:
            Memory snapshot data
        """
        return {
            'label': label,
            'timestamp': time.time(),
            'process_memory_mb': _get_process_memory(),
            'gpu_memory': _get_gpu_memory_usage(),
            'system_memory_percent': psutil.virtual_memory().percent if HAS_PSUTIL else None
        }


@contextlib.contextmanager
def managed_memory_context(auto_cleanup: bool = True, deep_clean: bool = False):
    """
    Context manager for automatic memory management.

    Args:
        auto_cleanup: Whether to perform cleanup on exit
        deep_clean: Whether to use aggressive GPU cleanup

    Usage:
        with managed_memory_context(deep_clean=True):
            # Train your models here
            pass
    """
    manager = MemoryManager()

    try:
        yield manager
    finally:
        if auto_cleanup:
            results = manager.cleanup_all(force=deep_clean)

            # Only print if significant memory was freed
            memory_freed = results.get('memory_freed_mb', 0)
            if memory_freed > 100:  # More than 100MB
                print(f"Memory cleanup freed {memory_freed:.0f}MB")


def suggest_gpu_batch_size(image_height: int, image_width: int,
                           channels: int = 3, available_memory_mb: int = 4096) -> int:
    """
    Suggest batch size based on image dimensions and available GPU memory.

    Args:
        image_height: Input image height in pixels
        image_width: Input image width in pixels
        channels: Number of image channels (1=grayscale, 3=RGB)
        available_memory_mb: Available GPU memory in MB

    Returns:
        Suggested batch size for the given constraints
    """
    # Estimate memory per image (input + activations + gradients)
    bytes_per_pixel = 4  # float32
    memory_multiplier = 6  # Rough estimate for activations/gradients in CNN

    bytes_per_image = image_height * image_width * channels * bytes_per_pixel * memory_multiplier
    mb_per_image = bytes_per_image / (1024 * 1024)

    # Use 70% of available memory for batch
    usable_memory = available_memory_mb * 0.7
    suggested_batch_size = max(1, int(usable_memory / mb_per_image))

    # Reasonable bounds
    return min(max(suggested_batch_size, 1), 128)