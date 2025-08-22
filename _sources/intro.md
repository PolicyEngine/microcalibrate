## MicroCalibrate

MicroCalibrate is a comprehensive framework for survey weight calibration that combines traditional calibration techniques with modern machine learning approaches. It enables users to adjust sample weights to match population targets while providing advanced features for sparsity, optimization, and robustness analysis.

## Key features

### 1. Core calibration
- **Survey weight adjustment**: The system calibrates sample weights to match known population totals using gradient-based optimization.
- **Multi-target support**: The calibration process can handle multiple calibration targets simultaneously.
- **Custom estimate functions**: Users can use either estimate matrices or custom functions for flexible calibration scenarios.

### 2. L0 regularization for sparsity
- **Dataset reduction**: The algorithm automatically identifies and zeros out unnecessary weights to reduce dataset size.
- **Sparse weight generation**: The system creates compact datasets while maintaining calibration accuracy.
- **Configurable sparsity**: Users can control the trade-off between dataset size and calibration precision.

### 3. Automatic hyperparameter tuning
- **Cross-validation**: The system uses holdout validation to find optimal regularization parameters.
- **Multi-objective optimization**: The optimization process balances calibration loss, accuracy, and sparsity.
- **Optuna integration**: The package leverages state-of-the-art hyperparameter optimization through Optuna.

### 4. Robustness evaluation
- **Generalization assessment**: The evaluation module assesses how well calibration performs on unseen targets.
- **Stability snalysis**: The system identifies targets that are difficult to calibrate reliably.
- **Actionable recommendations**: Users receive specific suggestions for improving calibration robustness.

### 5. Target analysis
- **Pre-calibration assessment**: The system identifies problematic targets before calibration begins.
- **Analytical solutions**: The analysis helps users understand the mathematical difficulty of target combinations.
- **Order of magnitude warnings**: The system detects targets that differ significantly from initial estimates.

### 6. Performance monitoring
- **Detailed logging**: The system tracks calibration progress across epochs.
- **Performance dashboard**: Users can visualize calibration results at https://microcalibrate.vercel.app/.
- **CSV export**: The system can save detailed performance metrics for further analysis.

## Dashboard requirements

To use the performance dashboard for visualization, users must ensure their calibration log CSV contains the following fields:
- epoch (int): The iteration number during calibration.
- target_name (str): The name of each calibration target.
- target (float): The target value to achieve.
- estimate (float): The estimated value at each epoch.
- error (float): The difference between target and estimate.
- abs_error (float): The absolute error value.
- rel_abs_error (float): The relative absolute error.
- loss (float): The loss value at each epoch.

## Getting started

```python
from microcalibrate import Calibration
import numpy as np
import pandas as pd

# Basic calibration
cal = Calibration(
    weights=initial_weights,
    targets=target_values,
    estimate_matrix=contribution_matrix
)
performance = cal.calibrate()

# With L0 regularization
cal = Calibration(
    weights=initial_weights,
    targets=target_values,
    estimate_matrix=contribution_matrix,
    regularize_with_l0=True
)
performance = cal.calibrate()
sparse_weights = cal.sparse_weights

# Hyperparameter tuning
best_params = cal.tune_l0_hyperparameters(n_trials=30)

# Robustness evaluation
robustness = cal.evaluate_holdout_robustness()
print(robustness['recommendation'])
```

## Documentation structure

- **[Basic calibration](calibration.ipynb)**: Core calibration concepts and basic usage
- **[L0 regularization](l0_regularization.ipynb)**: Creating sparse weights for dataset reduction
- **[Robustness evaluation](robustness_evaluation.ipynb)**: Assessing calibration stability and generalization

## Support

- **GitHub issues**: https://github.com/PolicyEngine/microcalibrate/issues
- **Documentation**: https://policyengine.github.io/microcalibrate/
- **Performance dashboard**: https://microcalibrate.vercel.app/