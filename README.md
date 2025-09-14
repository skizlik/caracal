# Caracal v0.0.1

**A Python library for rigorous analysis of machine learning model variability and performance stability.**

---

## Overview

Machine learning practitioners commonly train a model once and report its performance. However, stochastic training processes can produce significantly different outcomes across identical runs. **Caracal** provides tools to quantify and analyze this variability, helping researchers and practitioners understand model reliability and make more informed decisions.

> **Warning: Early Development Notice**: This is v0.0.1 - a working prototype with core functionality implemented. The API will evolve significantly before v0.1.0. Use for experimentation and feedback.

## What Works Now (v0.0.1)

- **Variability Studies**: Run identical models multiple times and analyze performance distributions
- **Statistical Analysis**: Compare model performance with effect sizes and significance tests
- **Model Wrappers**: Support for Keras/TensorFlow and scikit-learn models
- **Multiple Data Handlers**: TabularDataHandler, ImageDataHandler, TextDataHandler, TimeSeriesDataHandler
- **Visualization**: Specialized plots for training curves (neural networks) and run distributions (sklearn)
- **Enhanced Results**: Statistical integration and comprehensive analysis methods

## Quick Start

### Installation

```bash
git clone https://github.com/skizlik/caracal.git
cd caracal
pip install -e .
```

### GPU Support (Docker)
For TensorFlow GPU support in a containerized environment:
```bash
./build-gpu.sh    # Build the container
./run-gpu.sh      # Launch with GPU support
```

### Basic Usage

```python
import caracal as cr
from caracal import ModelConfig, TabularDataHandler

# Define your model builder
def create_model(config):
    # Your model creation logic
    return wrapped_model

# Set up data and configuration
data_handler = TabularDataHandler('data.csv', target_column='target')
config = ModelConfig({'epochs': 10, 'batch_size': 32})

# Run variability study
results = cr.run_variability_study(
    model_builder=create_model,
    data_handler=data_handler,
    model_config=config,
    num_runs=5
)

# Analyze results
print(results.summarize())
final_accuracies = results.get_final_metrics('val_accuracy')
```

## Current Architecture

### Core Components
- **ModelConfig**: Parameter management with validation
- **BaseModelWrapper**: Unified interface for different ML frameworks
- **ExperimentRunner**: Orchestrates variability studies
- **VariabilityStudyResults**: Enhanced results with analysis methods
- **DataHandler**: Configurable data loading and splitting

### Framework Support
- **Keras/TensorFlow**: Full support with epoch-by-epoch metrics
- **scikit-learn**: Adapted support with run-based variability analysis
- **Statistical Analysis**: Integration with scipy-based hypothesis tests

### Data Handlers (Available Now)
- **TabularDataHandler**: CSV files with automatic train/val/test splitting
- **ImageDataHandler**: Directory-based image classification datasets
- **TextDataHandler**: Text classification from CSV with tokenization
- **TimeSeriesDataHandler**: Time series data with sequence generation

## Planned for v0.1.0

- Hyperparameter sensitivity analysis
- Advanced convergence diagnostics
- Comprehensive documentation and tutorials
- Extended statistical analysis suite
- Performance optimization
- API stabilization

## Dependencies

**Required:**
- pandas, numpy, scikit-learn

**Optional (for full functionality):**
- tensorflow (neural network support, ImageDataHandler, TextDataHandler, TimeSeriesDataHandler)
- scipy (statistical tests)
- matplotlib, seaborn (plotting)
- mlflow (experiment tracking)
- shap (model explainability)
- hyperopt (hyperparameter tuning)

## Development Status

**Working**: Core variability studies, statistical analysis, visualization, data handlers
**Experimental**: sklearn integration, advanced plotting features, hyperparameter tuning
**Planned**: Comprehensive tutorials, API stabilization, performance optimization

## Contributing

This project is in active development. Feedback on the core architecture and API design is particularly valuable at this stage. Please open issues to discuss:

- Use cases and feature requirements
- API design preferences
- Integration with existing workflows
- Performance and scalability concerns

## License

MIT License - see LICENSE file for details.

---

*Caracal: **Computational Analysis of Run And Convergence Algorithms Library***
