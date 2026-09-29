"""Regression test for the sparse tracking precedence bug (finding #4).

The original expression ``i % tracking_n / 2 == 0`` parses as
``(i % tracking_n) / 2 == 0``, which is equivalent to
``i % tracking_n == 0`` -- the ``/ 2`` is a no-op. The intent was to
log sparse-loop estimates twice as often as the dense loop (because
the sparse loop runs 2x as many epochs). After the fix the sparse
performance DataFrame has roughly 2x the row density of the dense one.
"""

import logging

import numpy as np
import pandas as pd
import torch

from microcalibrate.reweight import reweight


def test_sparse_loop_logs_twice_as_often_as_dense() -> None:
    """With the fix in place, the sparse performance DataFrame should
    contain approximately twice as many tracked epoch rows per target
    as the dense DataFrame for the same number of dense epochs.
    """

    def estimate_function(w: torch.Tensor) -> torch.Tensor:
        return w.sum().unsqueeze(0)

    rng = np.random.default_rng(0)
    original_weights = rng.uniform(1.0, 2.0, size=50)
    targets = np.array([original_weights.sum() * 1.02])
    logger = logging.getLogger("test_sparse_tracking")

    torch.manual_seed(0)
    _w, _sparse, dense_df = reweight(
        original_weights=original_weights,
        estimate_function=estimate_function,
        targets_array=targets,
        target_names=np.array(["total"]),
        l0_lambda=0.0,
        init_mean=0.999,
        temperature=0.5,
        regularize_with_l0=False,
        sparse_learning_rate=0.2,
        dropout_rate=0.0,
        epochs=100,
        noise_level=0.0,
        learning_rate=1e-3,
        logger=logger,
    )

    # Rerun with L0 enabled to capture the sparse performance DF. The
    # sparse performance DataFrame is written to the
    # <csv>_sparse.csv path when csv_path is set; we don't need disk
    # here -- instead we count rows on the dense performance_df and
    # compare indirectly by tracking the epoch lists via csv output.

    # To keep the assertion simple and robust to refactors, we verify
    # the sparse_tracking_n itself via a minimal L0 run and inspection
    # of the resulting CSVs. We exercise the L0 path by writing to a
    # temporary CSV and reading the sparse file back.
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "out.csv")
        torch.manual_seed(0)
        reweight(
            original_weights=original_weights,
            estimate_function=estimate_function,
            targets_array=targets,
            target_names=np.array(["total"]),
            l0_lambda=1e-6,
            init_mean=0.999,
            temperature=0.5,
            regularize_with_l0=True,
            sparse_learning_rate=0.01,
            dropout_rate=0.0,
            epochs=100,
            noise_level=0.0,
            learning_rate=1e-3,
            csv_path=csv_path,
            logger=logger,
        )
        sparse_df = pd.read_csv(csv_path.replace(".csv", "_sparse.csv"))

    # With epochs=100, tracking_n = max(1, 100 // 10) = 10. Dense loop
    # logs at i in {0, 10, 20, ..., 90} => 10 tracked epochs.
    # Sparse loop runs 2*100 = 200 epochs at stride max(1, 10 // 2) = 5
    # => i in {0, 5, 10, ..., 195} => 40 tracked epochs. Per target the
    # DataFrame therefore has ~4x the row count (2x density * 2x
    # epochs). Under the previous bug the sparse stride was 10 giving
    # ~20 tracked epochs (2x only).
    n_targets = 1
    dense_rows_per_target = len(dense_df) / n_targets
    sparse_rows_per_target = len(sparse_df) / n_targets
    ratio = sparse_rows_per_target / max(dense_rows_per_target, 1)
    assert ratio >= 3.5, (
        f"Sparse tracking density ratio is {ratio:.2f} (dense rows "
        f"={dense_rows_per_target}, sparse rows={sparse_rows_per_target}); "
        "expected ~4x after fixing the precedence bug."
    )
