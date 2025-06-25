import numpy as np
import pandas as pd
import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def analyse_targets(
    weights,
    estimate_matrix,
    targets_array,
):
    target_names = np.array(estimate_matrix.columns)
    estimate_matrix = torch.tensor(
        estimate_matrix.values, dtype=torch.float32, device=device
    )
    targets_array = torch.tensor(
        targets_array, dtype=torch.float32, device=device
    )
    weights = torch.tensor(
        np.log(weights), requires_grad=True, dtype=torch.float32, device=device
    )

    # Compute estimates
    estimate = torch.exp(weights) @ estimate_matrix

    # Compute relative errors for each target
    rel_errors = ((estimate - targets_array) / targets_array) ** 2

    # Use jacobian to compute gradients for all targets at once
    gradients = torch.autograd.functional.jacobian(
        func=lambda w: (
            (torch.exp(w) @ estimate_matrix - targets_array) / targets_array
        )
        ** 2,
        inputs=weights,
    )

    # gradients is now [n_targets, n_weights]
    gradients_numpy = gradients.cpu().numpy()
    gradients_df = pd.DataFrame(gradients_numpy, index=target_names)

    # Calculate correlation matrix
    target_correlation = gradients_df.T.corr().mean()

    df = pd.DataFrame(
        {
            "target": target_names,
            "average_correlation": target_correlation,
            "relative_error": rel_errors.detach().cpu().numpy() ** 0.5,
        }
    )

    return df
