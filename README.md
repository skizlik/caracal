# Caracal

**A Python library for the rigorous analysis of machine learning model stability and performance.**

---

## Caracal: Computational Analysis of Run And Convergence Algorithms Library

A common practice in applied machine learning is to train a model once and report its final performance metric. However, the stochastic nature of the training process means that identical runs can produce a significant range of outcomes. `Caracal` is a project to build a lightweight, intuitive toolkit for quantifying the stability, reliability, and hyperparameter sensitivity of ML models.

This is an active, early-stage development project. The core architecture is being built, but the API is subject to change, and many features are not yet fully implemented or tested. The primary goal is to build a robust foundation for a powerful, professional-grade tool.

## Key Features (Under Development)

The goal of the `v0.1.0` release is to include:
* **Model Agnostic Wrappers** for Keras and Scikit-learn.
* **Stability Analysis** to analyze the statistical distribution of performance metrics.
* **Sensitivity Analysis** to automate testing across a range of hyperparameters.
* **Training Diagnostics** to quantify "bounciness" and detect "model collapse."

## Installation (for Development & Testing)

`Caracal` is not yet available on PyPI. To test the current development version, clone this repository and install it in "editable mode":

```bash
git clone [https://github.com/skizlik/caracal.git](https://github.com/skizlik/caracal.git)
cd caracal
pip install -e .
```

## Contributing
Feedback, ideas, and contributions at this early stage are highly welcome. If you are interested in the problem of model stability, please feel free to open an issue to discuss your thoughts.

## License
This project is licensed under the terms of the **MIT License**.
