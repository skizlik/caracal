# Caracal

[![Tests](https://github.com/skizlik/caracal/workflows/Tests/badge.svg)](https://github.com/skizlik/caracal/actions)
[![Coverage](https://img.shields.io/badge/coverage-31%25-yellow)](https://github.com/skizlik/caracal)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.0.4-blue)](https://github.com/skizlik/caracal/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/skizlik/caracal/blob/main/LICENSE)

**Caracal** (Comparison And Reproducibility Analysis for Computational Algorithmic Learning) is a Python library for analyzing variability, reproducibility, and stability in machine learning models.

## 🎯 The Problem Caracal Solves

Your model achieves 95% accuracy. But is it *really* reliable?
```python
# Without Caracal: Deploy and hope for the best
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")  # 95% - ship it!

# With Caracal: Know your model's true reliability
results = cr.run_variability_study(model_builder, data_handler, config, num_runs=10)
stability = cr.assess_training_stability(results.all_runs_metrics)
print(f"Accuracy: {results.mean_accuracy:.2%} ± {results.std_accuracy:.2%}")  # 95% ± 8% 😱
print(f"Stability: {stability['stability_assessment']}")  # "LOW" - don't ship!
```

## ✨ Key Features

### 🔬 **Statistical Rigor**
- **Publication-ready statistical tests** with effect sizes and assumption checking
- Mann-Whitney U, Wilcoxon signed-rank, ANOVA, Kruskal-Wallis tests
- Multiple comparison corrections (Holm, Bonferroni, FDR)
- Automatic assumption validation with recommendations

### 📊 **Variability Analysis**
- Run multiple training iterations to measure model stability
- Detect initialization-dependent performance variations
- Identify convergence issues and training instabilities
- Assess reproducibility across different random seeds

### 🧹 **Memory Management**
- **Automatic GPU memory cleanup** prevents OOM errors
- Optional process isolation for guaranteed cleanup
- Memory monitoring and reporting
- Smart resource management for long-running experiments

### 📈 **Comprehensive Visualization**
- Training history plots with confidence bands
- Statistical comparison matrices
- Stability assessment dashboards
- Distribution plots for final metrics

## 🚀 Quick Start

### Installation
```bash
# Basic installation
pip install caracal

# With all optional dependencies
pip install caracal[all]

# Development installation
git clone https://github.com/skizlik/caracal.git
cd caracal
pip install -e .
```

### Basic Usage
```python
import caracal as cr
from caracal import ModelConfig, run_variability_study

# 1. Define your model configuration
config = ModelConfig({
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001
})

# 2. Create a model builder function
def create_model(config):
    model = build_your_model()  # Your model creation logic
    return cr.KerasModelWrapper(model)

# 3. Set up your data
data_handler = cr.TabularDataHandler('data.csv', target_column='label')

# 4. Run variability study
results = run_variability_study(
    model_builder=create_model,
    data_handler=data_handler,
    model_config=config,
    num_runs=10
)

# 5. Analyze results
print(results.summarize())

# 6. Statistical comparison
comparison = results.compare_models_statistically('val_accuracy')
print(f"Runs are statistically different: {comparison['overall_test'].is_significant()}")

# 7. Visualize
cr.plot_variability_summary(results.all_runs_metrics, results.final_val_accuracies)
```

## 📚 Examples

### Comparing Model Architectures
```python
# Define different architectures
architectures = {
    'shallow': create_shallow_model,
    'deep': create_deep_model,
    'regularized': create_regularized_model
}

# Run studies for each
results_dict = {}
for name, builder in architectures.items():
    results = cr.run_variability_study(builder, data_handler, config)
    results_dict[name] = results.final_val_accuracies

# Statistical comparison with publication-ready output
comparison = cr.compare_multiple_models(results_dict)
print(cr.generate_statistical_summary([comparison['overall_test']]))
```

### Memory-Safe Training
```python
# Process isolation for guaranteed cleanup
with cr.managed_memory(use_process_isolation=True):
    for i in range(100):  # Train 100 models without memory leaks
        model = create_large_model()
        model.fit(data)
        # Memory automatically cleaned after each iteration
```

## 🛠️ Development Status

**Current Version: 0.0.4** (Pre-release)

This is an active research project under rapid development. The API may change between versions until v1.0.0.

### What's Working
- ✅ Core variability analysis
- ✅ Statistical tests with effect sizes
- ✅ Memory management
- ✅ Basic plotting functions
- ✅ 68 automated tests (31% coverage)

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=caracal --cov-report=term-missing

# Run specific test module
pytest tests/test_config.py -v
```

## 🐳 Docker Development

For GPU-enabled development with all dependencies:
```bash
# Build the Docker image
./build-gpu.sh

# Run Jupyter Lab
./run-gpu.sh

# Run tests in container
./run-gpu.sh pytest tests/
```

## 📖 Documentation

Full documentation is in development. For now:
- See `examples/` for Jupyter notebooks
- Check docstrings for detailed API documentation
- Review tests for usage patterns

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/caracal.git
cd caracal

# Install in development mode
pip install -e .[dev]

# Run tests to verify setup
pytest tests/
```

## 📊 Feature Availability

Caracal has modular dependencies. Core functionality always works, with optional features available when dependencies are installed:
```python
import caracal
caracal.print_feature_summary()  # See what's available
```

| Feature | Required Package | Install Command |
|---------|-----------------|-----------------|
| TensorFlow/Keras Support | tensorflow | `pip install tensorflow` |
| Advanced Statistics | scipy | `pip install scipy` |
| Plotting | matplotlib, seaborn | `pip install matplotlib seaborn` |
| MLflow Integration | mlflow | `pip install mlflow` |
| Model Explainability | shap | `pip install shap` |
| Hyperparameter Tuning | hyperopt | `pip install hyperopt` |

## 📝 Citation

If you use Caracal in your research, please cite:
```bibtex
@software{caracal2025,
  title = {Caracal: A Framework for ML Model Variability Analysis},
  author = {Kizlik, Stephen},
  year = {2025},
  url = {https://github.com/skizlik/caracal},
  version = {0.0.4}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to all early users and contributors who are helping shape Caracal's development.

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/skizlik/caracal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/skizlik/caracal/discussions)
- **Author**: Stephen Kizlik (stephen.kizlik@gmail.com)

---

**Note**: Caracal is under active development. We recommend pinning to specific versions in production:
```
caracal==0.0.4
```
