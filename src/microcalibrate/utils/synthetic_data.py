import numpy as np
import pandas as pd


def simulate_contradictory_data(
    T: float,  # official grand total
    k: int,  # number of strata
    c: float,  # contradiction factor. Strata totals fall short by floor(c * T)
    n: int,  # Sample size, over all strata
    dirichlet_alpha=5.0,  # low value creates uneven divisions
    gamma_shape: float = 2.0,
    gamma_scale: float = 2.0,
    seed: int = None,
):
    """Returns tuple containing the sample DataFrame and the official totals dict."""
    rng = np.random.default_rng(seed)

    # Simulate stratum proportions (thetas)
    alphas = np.full(k, dirichlet_alpha)
    thetas = rng.dirichlet(alphas)

    # Define Stratum subtotals S_i
    S_i = thetas * T

    # Calculate and distribute the contradiction 'delta'
    delta = np.floor(c * T)
    p = np.full(k, 1 / k)
    delta_i = rng.multinomial(n=int(delta), pvals=p)

    # Calculate final, contradictory official subtotals S_i^*
    S_i_star = S_i - delta_i

    # Allocate sample size n into strata sample sizes n_i
    n_i_float = (
        thetas * n
    )  # Assuming sample size in proportion to stratum total
    n_i = n_i_float.astype(int)
    remainder_n = n - n_i.sum()
    if remainder_n > 0:
        top_indices_n = np.argsort(n_i_float - n_i)[-remainder_n:]
        n_i[top_indices_n] += 1
    if n_i.sum() != n:
        diff = n_i.sum() - n
        while diff > 0:
            largest_idx = np.argmax(n_i)
            if n_i[largest_idx] > 1:
                n_i[largest_idx] -= 1
                diff -= 1
            else:
                break

    # Simulate population microdata, draw sample, and calculate weights
    all_samples = []
    for i in range(k):
        sample_y = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=n_i[i])
        weight = np.full(n_i[i], S_i_star[i] / np.sum(sample_y))  # baseline
        stratum_sample = pd.DataFrame(
            {"stratum_id": i + 1, "y_ij": sample_y, "w_ij": weight}
        )
        all_samples.append(stratum_sample)

    # Combine and return
    final_sample_df = pd.concat(all_samples, ignore_index=True)
    official_totals = {"T_official": T, "S_star_official": S_i_star}
    return final_sample_df, official_totals
