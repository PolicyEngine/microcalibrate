# MicroCalibrate

[![CI](https://github.com/PolicyEngine/microcalibrate/actions/workflows/main.yml/badge.svg)](https://github.com/PolicyEngine/microcalibrate/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/PolicyEngine/microcalibrate/branch/main/graph/badge.svg)](https://codecov.io/gh/PolicyEngine/microcalibrate)
[![PyPI version](https://badge.fury.io/py/microcalibrate.svg)](https://badge.fury.io/py/microcalibrate)
[![Python Version](https://img.shields.io/pypi/pyversions/microcalibrate)](https://pypi.org/project/microcalibrate/)

MicroCalibrate is a Python package for calibrating survey weights to match population targets, with advanced features including L0 regularization for sparsity, hyperparameter tuning, and robustness evaluation.

## Features

- **Survey Weight Calibration**: The package adjusts sample weights to match known population totals.
- **L0 Regularization**: The system creates sparse weights to reduce dataset size while maintaining accuracy.
- **Automatic Hyperparameter Tuning**: The optimization module automatically finds optimal regularization parameters using cross-validation.
- **Robustness Evaluation**: The evaluation tools assess calibration stability using holdout validation.
- **Target Assessment**: The analysis features help identify which targets complicate calibration.
- **Performance Monitoring**: The system tracks calibration progress with detailed logging.
- **Interactive Dashboard**: Users can visualize calibration performance at https://microcalibrate.vercel.app/.

## Installation

```bash
pip install microcalibrate
```

The package requires the following dependencies:
- Python version 3.13 or higher is required.
- PyTorch version 2.7.0 or higher is needed.
- Additional required packages include NumPy, Pandas, Optuna, and L0-python.

## Quick start

### Basic calibration

```python
from microcalibrate import Calibration
import numpy as np
import pandas as pd

# Create sample data for calibration
n_samples = 1000
weights = np.ones(n_samples)  # Initial weights are set to one

# Create an estimate matrix that represents the contribution of each record to targets
estimate_matrix = pd.DataFrame({
    'total_income': np.random.normal(50000, 15000, n_samples),
    'total_employed': np.random.binomial(1, 0.6, n_samples),
})

# Set the target values to achieve through calibration
targets = np.array([
    50_000_000,  # This is the total income target
    600,         # This is the total employed target
])

# Initialize the calibration object and configure the optimization parameters
cal = Calibration(
    weights=weights,
    targets=targets,
    estimate_matrix=estimate_matrix,
    epochs=500,
    learning_rate=1e-3,
)

# Perform the calibration to adjust weights
performance_df = cal.calibrate()

# Retrieve the calibrated weights from the calibration object
new_weights = cal.weights
```

## API reference

### Calibration class

The Calibration class is the main class for weight calibration.

**Parameters:**
- `weights`: The initial weights array for each record.
- `targets`: The target values to match during calibration.
- `estimate_matrix`: A DataFrame containing the contribution of each record to targets.
- `estimate_function`: An alternative to estimate_matrix that uses a custom function.
- `epochs`: The number of optimization iterations to perform (default is 32).
- `learning_rate`: The optimization learning rate (default is 1e-3).
- `noise_level`: The amount of noise added for robustness (default is 10.0).
- `dropout_rate`: The dropout rate for regularization (default is 0).
- `regularize_with_l0`: This parameter enables L0 regularization (default is False).
- `l0_lambda`: The L0 regularization strength parameter (default is 5e-6).
- `init_mean`: The initial proportion of non-zero weights (default is 0.999).
- `temperature`: The sparsity control parameter (default is 0.5).

**Methods:**
- `calibrate()`: This method performs the weight calibration process.
- `tune_l0_hyperparameters()`: This method automatically tunes L0 parameters using cross-validation.
- `evaluate_holdout_robustness()`: This method assesses calibration stability using holdout validation.
- `assess_analytical_solution()`: This method analyzes the difficulty of achieving target combinations.
- `summary()`: This method returns a summary of the calibration results.

## Examples and documentation

For detailed examples and interactive notebooks, see the [documentation](https://policyengine.github.io/microcalibrate/).

## Contributing

Contributions are welcome to the project. Please feel free to submit a Pull Request with your improvements.
