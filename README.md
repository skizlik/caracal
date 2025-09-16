# Caracal v0.0.2

**Computational Analysis of Run And Convergence Algorithms Library**

*A Python library for rigorous analysis of machine learning model variability and performance stability.*

---

## Overview

Machine learning practitioners commonly train a model once and report its performance. However, stochastic training processes can produce significantly different outcomes across identical runs. **Caracal** provides tools to quantify and analyze this variability, helping researchers and practitioners understand model reliability and make more informed decisions.

## What's New in v0.0.2

**Major Reliability Improvements**
- **Fixed GPU memory management** - Completely refactored memory system for reliable repeated training
- **Eliminated memory leaks** - GPU-intensive models can now run 50+ training iterations without errors
- **Clean output** - Removed parameter spam and verbose logging for better user experience
- **Professional GPU handling** - Proper TensorFlow configuration and cleanup between runs

v0.0.2 represents a major step forward in stability, enabling real-world variability studies on large models.

## What Works Now

- **Variability Studies**: Run identical models multiple times and analyze performance distributions
- **Statistical Analysis**: Compare model performance with effect sizes and significance tests  
- **Reliable GPU Training**: Handle large CNNs and repeated training runs without memory issues
- **Model Wrappers**: Support for Keras/TensorFlow and scikit-learn models
- **Multiple Data Handlers**: Tabular, Image, Text, and Time Series data support
- **Enhanced Memory Management**: Automatic cleanup with deep GPU cleaning for intensive workloads
- **Visualization**: Training curves, performance distributions, and statistical plots
- **Statistical Integration**: Comprehensive analysis methods with effect sizes

## Installation

```bash
git clone https://github.com/skizlik/caracal.git
cd caracal
pip install -e .
```

### Essential Dependencies

```bash
# Core functionality
pip install pandas numpy scikit-learn

# For neural networks and GPU support
pip install tensorflow

# For statistical analysis and plotting
pip install scipy matplotlib seaborn
```

## Quick Start Example

Here's a complete working example for a tabular dataset:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import caracal as cr
from caracal import ModelConfig
from caracal.data import TabularDataHandler
from caracal.core import ScikitLearnModelWrapper

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
data['target'] = y
data.to_csv('sample_data.csv', index=False)

# Define model builder function
def create_rf_model(config):
    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 10),
        random_state=config.get('random_state', 42)
    )
    return ScikitLearnModelWrapper(model, model_id="random_forest")

# Set up data handler and configuration
data_handler = TabularDataHandler('sample_data.csv', target_column='target')
config = ModelConfig({
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
})

# Run variability study
results = cr.run_variability_study(
    model_builder=create_rf_model,
    data_handler=data_handler,
    model_config=config,
    num_runs=5
)

# Analyze results
print(results.summarize())

# Get statistical comparison
final_accuracies = results.get_final_metrics('val_accuracy')
print(f"Final validation accuracies: {final_accuracies}")
```

## GPU Memory Management

For GPU-intensive workloads (large CNNs, many training runs):

```python
from caracal.memory import set_tensorflow_env_vars, setup_tensorflow_gpu, managed_memory_context

# Configure GPU before importing TensorFlow
set_tensorflow_env_vars()

# Set up GPU memory management  
setup_tensorflow_gpu(memory_limit_mb=4096)  # Optional memory limit

# Use deep cleanup for intensive workloads
with managed_memory_context(deep_clean=True):
    results = cr.run_variability_study(
        model_builder=create_cnn_model,
        data_handler=image_data_handler,
        model_config=config,
        num_runs=50  # Now reliable for large studies
    )
```

## CNN Example

For image classification with convolutional neural networks:

```python
import tensorflow as tf
from caracal import ModelConfig
from caracal.data import ImageDataHandler
from caracal.core import KerasModelWrapper
from caracal.memory import managed_memory_context

def create_cnn_model(config):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', 
                              input_shape=config['input_shape']),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(config['num_classes'], activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return KerasModelWrapper(model, model_id="cnn")

# Set up for image data (assumes image_dataset/ directory structure)
image_handler = ImageDataHandler('image_dataset/', image_size=(128, 128))
cnn_config = ModelConfig.for_cnn(input_shape=(128, 128, 3), num_classes=10)

# Run with proper memory management
with managed_memory_context(deep_clean=True):
    results = cr.run_variability_study(
        model_builder=create_cnn_model,
        data_handler=image_handler,
        model_config=cnn_config,
        num_runs=10
    )
```

## Architecture

### Core Components

- **ModelConfig**: Parameter management with validation and smart defaults
- **BaseModelWrapper**: Unified interface for different ML frameworks
- **ExperimentRunner**: Orchestrates variability studies with memory management
- **VariabilityStudyResults**: Enhanced results container with analysis methods
- **MemoryManager**: GPU-focused memory management for reliable repeated training

### Framework Support

- **Keras/TensorFlow**: Full support with epoch-by-epoch metrics and GPU memory management
- **scikit-learn**: Adapted support with run-based variability analysis
- **Statistical Analysis**: Integration with scipy-based hypothesis tests and effect sizes

### Data Handlers

- **TabularDataHandler**: CSV files with train/val/test splitting
- **ImageDataHandler**: Directory-based image datasets with TensorFlow integration
- **TextDataHandler**: Text classification with tokenization and padding
- **TimeSeriesDataHandler**: Time series with sequence generation

## Memory Management Features

Caracal includes sophisticated memory management designed for ML workloads:

- **Automatic cleanup** between training runs
- **GPU memory optimization** for TensorFlow models
- **Deep cleaning** for intensive workloads (50+ training runs)
- **Environment variable configuration** for common GPU issues
- **Memory usage monitoring** and reporting

## GPU and Docker Support

### Docker with GPU Support
```bash
./build-gpu.sh    # Build container with CUDA support
./run-gpu.sh      # Launch with GPU access
```

### Manual GPU Configuration
```python
# Set environment variables before importing TensorFlow
from caracal.memory import set_tensorflow_env_vars
set_tensorflow_env_vars()

# Configure GPU memory growth
from caracal.memory import setup_tensorflow_gpu  
setup_tensorflow_gpu()
```

## Statistical Analysis

Compare model performance with proper statistical methods:

```python
# Statistical comparison of multiple runs
comparison = results.compare_models_statistically('val_accuracy')
print(comparison['overall_test'].conclusion)

# Compare multiple algorithms
algorithms = ['random_forest', 'svm', 'neural_net']
all_results = {}

for alg in algorithms:
    results = cr.run_variability_study(
        model_builder=lambda config: create_model(alg, config),
        data_handler=data_handler,
        model_config=config,
        num_runs=10
    )
    all_results[alg] = results.get_final_metrics('val_accuracy')

# Statistical comparison across algorithms
from caracal.analysis import compare_multiple_models
comparison = compare_multiple_models(all_results)
```

## Development Status

**Stable (v0.0.2)**: Core variability studies, memory management, basic statistical analysis

**Working**: Model wrappers, data handlers, visualization, GPU memory management

**Experimental**: Advanced statistical features, hyperparameter tuning

**Planned for v0.1.0**: API stabilization, comprehensive documentation, performance optimization

## Dependencies

**Core Requirements:**
- `pandas`, `numpy`, `scikit-learn`

**Framework Support:**
- `tensorflow` (neural networks, image/text/timeseries data handlers)
- `scipy` (statistical tests and effect sizes)

**Visualization:**
- `matplotlib`, `seaborn` (plotting functions)

**Optional Features:**
- `mlflow` (experiment tracking)
- `psutil` (memory monitoring)
- `shap` (model explainability)
- `hyperopt` (hyperparameter optimization)

## Contributing

This project is in active development. The v0.0.2 release represents a major stability milestone, but feedback is still valuable for API design and feature prioritization.

**Areas for feedback:**
- Use cases and workflow integration
- API design and ease of use
- Performance with your specific models/datasets
- Additional statistical analysis needs

## License

MIT License - see LICENSE file for details.

---

*Caracal: **Computational Analysis of Run And Convergence Algorithms Library***

**Repository**: https://github.com/skizlik/caracal