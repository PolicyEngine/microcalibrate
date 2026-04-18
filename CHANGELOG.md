## [0.22.0] - 2026-04-18

### Added

- Added `batch_size` parameter to `Calibration` and `reweight()` for gradient accumulation over record batches. When set, the chi-squared loss is accumulated under `no_grad` in a first pass and the backward pass is split into per-batch virtual-loss calls with pre-computed per-target coefficients. Peak autograd activation memory drops from O(n_records × n_targets) to O(batch_size × n_targets). The full-batch path is unchanged when `batch_size` is `None` (default) or greater than or equal to `n_records`. Not supported in combination with `regularize_with_l0=True` (raises `ValueError`).

### Changed

- `Calibration` now converts the user-provided `estimate_matrix` DataFrame to a cached `float32` torch tensor on `estimate_matrix_tensor` during `__init__` and releases the pandas DataFrame reference by setting `original_estimate_matrix` to `None`. Downstream code (`hyperparameter_tuning`, `evaluation`, `assess_analytical_solution`) reads the cached tensor rather than re-materializing from `DataFrame.values`. This substantially reduces peak RSS during `calibrate()` at large record counts. External readers of `Calibration.original_estimate_matrix` will now see `None` after construction; the tensor equivalent is available on `Calibration.estimate_matrix_tensor`.


## [0.21.3] - 2026-04-18

### Fixed

- Collapse duplicate device-init block in Calibration.__init__. The second copy re-ran the fallback chain, contained an unreachable else branch, and checked `self.device == "cuda"` (a torch.device vs string comparison that never triggered CUDA seeding on the default code path). torch is now seeded uniformly on every path and `torch.cuda.manual_seed_all` is invoked whenever `self.device.type == "cuda"`.
- Reimplement dropout_weights() correctly for log-space weights. The previous implementation set masked log-entries to 0 (exp(0) = 1, not dropped) and divided by the sum of logs, which on realistic weight scales could cross zero and inject Inf/NaN into training. The new implementation applies standard inverted dropout: dropped entries go to -inf in log space (and therefore zero in linear space), survivors are scaled by 1/(1-p) so the expected linear-space sum is preserved.
- Fix holdout state restoration and seed-namespace collision. `evaluate_holdout_robustness` and `tune_l0_hyperparameters` only restored captured attributes whose value was not None, so `excluded_targets=None` callers had the last holdout set's target list silently stick to the calibrator. Restoration is now unconditional. Robustness evaluation now uses `seed + 10_000` when generating its holdout sets instead of `seed + 1`, avoiding the deterministic index-aligned collision with tuning's holdout sets that leaked tuning data into "independent" evaluation.
- Guard the `(target + 1)` denominator in loss() and pct_close() against targets equal to -1 (divide-by-zero -> Inf) and extend the guard in loss() to reject non-finite rel_error (not just NaN). For any target with target+1 comfortably positive the numeric result is unchanged.
- Seed the NumPy RNG used for initial weight noise in reweight(). Previously only torch was seeded, so two Calibration runs with the same seed produced different initial log-weights (and therefore different trajectories), breaking the documented reproducibility guarantee. reweight() now accepts an explicit seed parameter, uses a local numpy.random.default_rng so the caller's global state is not mutated, and Calibration threads its seed through.
- Fix three latent bugs in reweight(): (1) the final dense-epoch gradient step was being silently skipped due to an off-by-one guard, so the returned weights were inconsistent with the tracked estimates; (2) np.log(original_weights) produced -inf and poisoned gradients when any initial weight was zero; and (3) the sparse L0 loop raised ZeroDivisionError in its tqdm postfix when start_loss happened to be zero.
- Fix operator-precedence bug in the sparse L0 tracking condition. `i % tracking_n / 2 == 0` parsed as `(i % tracking_n) / 2 == 0`, which is equivalent to `i % tracking_n == 0`, so the intended 2x logging density was silently lost. The sparse loop now logs at stride `max(1, tracking_n // 2)` so its tracked DataFrame has ~2x the row density of the dense loop, matching the 2x epoch count.
- Migrated versioning workflow from expired `POLICYENGINE_GITHUB` PAT to a short-lived GitHub App token, matching the pattern used by `policyengine-us`, `policyengine-core`, and `microdf`.


## [0.21.2] - 2026-02-24

### Changed

- Migrated from changelog_entry.yaml to towncrier fragments to eliminate merge conflicts.


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.21.1] - 2026-01-06 00:04:35

### Fixed

- Fixed critical React Server Components CVE vulnerabilities (CVE-2025-55182, CVE-2025-66478)

## [0.21.0] - 2025-08-22 13:47:07

### Added

- Adding documentation for L0, hyperparameter tuning and robustness checks.

## [0.20.0] - 2025-08-22 11:04:35

### Added

- Add hyperparameter tuning for L0 implementation with option to holdout targets.
- Add method to evaluate robustness of calibration to target holdouts.

## [0.19.2] - 2025-08-22 07:42:30

### Changed

- Moved to PolicyEngine's L0 package for regularization implementation.
- Moved to python 3.13.

## [0.19.1] - 2025-08-11 15:25:07

### Added

- Added a parameter to adjust the learning rate of the sparse optimizer.
- Fixed label in dashboard that incorrectly displayed 'estimate' instead of 'target'.

## [0.19.0] - 2025-08-04 15:53:14

### Added

- Adding column to sort by calibration difference to dashboard comparisons page.
- Ensure pagination so all repo branches are visible in the github loading.

## [0.18.0] - 2025-07-25 14:42:01

### Added

- A function to evaluate whether estimates are within desired tolerance levels.

## [0.17.0] - 2025-07-25 13:26:12

### Added

- L0 regularization logic.

## [0.16.0] - 2025-07-21 16:09:24

### Added

- Adding analytical assessment of targets to Calibration class.
- Enhance dashboard to show all targets even if not overlapping and better font / table view.

## [0.15.0] - 2025-07-15 10:32:10

### Added

- Add excluded_targets logic to handle holdout targets.

## [0.14.1] - 2025-07-07 12:34:47

### Changed

- Normalization parameter to handle multi-level geography calibration added to Calibration class.

## [0.14.0] - 2025-07-07 11:51:26

### Added

- Normalization parameter to handle multi-level geography calibration.

## [0.13.5] - 2025-06-30 17:15:57

### Changed

- Taking abs val for abs_rel_error denominator.

## [0.13.4] - 2025-06-30 15:30:19

### Changed

- Increase limit to csv size.

## [0.13.3] - 2025-06-30 13:40:33

### Changed

- Subsample to 10 epochs when loading dashboard.

## [0.13.2] - 2025-06-26 11:47:53

### Changed

- Loading dashboard automatically when sharing a deeplink.

## [0.13.1] - 2025-06-26 11:41:39

### Fixed

- Final weights are now consistent with the training log.

## [0.13.0] - 2025-06-26 11:01:53

### Added

- Adding total loss and error over epoch plots to dashboard.
- Ordering targets alphanumerically in dashboard.

## [0.12.0] - 2025-06-25 17:27:46

### Added

- Creating deeplinks.

## [0.11.0] - 2025-06-25 14:58:22

### Added

- Adding GitHub artifact comparison to dashboard.

## [0.10.0] - 2025-06-25 14:31:58

### Added

- Estimate function, over loss matrix.

## [0.9.0] - 2025-06-24 16:38:27

### Added

- Small performance dashboard fix.

## [0.8.0] - 2025-06-24 13:08:56

### Added

- Creating github artifact to save calibration log for test.
- Interface to load CSVs from GitHub.

## [0.7.0] - 2025-06-24 10:25:50

### Added

- Adding the calibration performance dashboard link to documentation.

## [0.6.0] - 2025-06-24 09:23:16

### Added

- Calibration performance dashboard.

## [0.5.0] - 2025-06-23 10:15:54

### Added

- Test for warning logic in Calibration() input checks.

## [0.4.0] - 2025-06-20 12:55:18

### Added

- Summary of calibration results.

## [0.3.0] - 2025-06-20 11:03:38

### Added

- Logging performance across epochs when calibrating.

## [0.2.0] - 2025-06-19 16:34:04

### Added

- Basic Calibration input checks.

## [0.1.0] - 2025-06-18 13:44:19

### Added

- Initialized project.

## [0.1.0] - 2025-06-18 13:19:30

### Changed

- Initialized changelogging.



[0.21.1]: https://github.com/PolicyEngine/microcalibrate/compare/0.21.0...0.21.1
[0.21.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.20.0...0.21.0
[0.20.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.19.2...0.20.0
[0.19.2]: https://github.com/PolicyEngine/microcalibrate/compare/0.19.1...0.19.2
[0.19.1]: https://github.com/PolicyEngine/microcalibrate/compare/0.19.0...0.19.1
[0.19.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.18.0...0.19.0
[0.18.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.17.0...0.18.0
[0.17.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.16.0...0.17.0
[0.16.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.15.0...0.16.0
[0.15.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.14.1...0.15.0
[0.14.1]: https://github.com/PolicyEngine/microcalibrate/compare/0.14.0...0.14.1
[0.14.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.13.5...0.14.0
[0.13.5]: https://github.com/PolicyEngine/microcalibrate/compare/0.13.4...0.13.5
[0.13.4]: https://github.com/PolicyEngine/microcalibrate/compare/0.13.3...0.13.4
[0.13.3]: https://github.com/PolicyEngine/microcalibrate/compare/0.13.2...0.13.3
[0.13.2]: https://github.com/PolicyEngine/microcalibrate/compare/0.13.1...0.13.2
[0.13.1]: https://github.com/PolicyEngine/microcalibrate/compare/0.13.0...0.13.1
[0.13.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.9.0...0.10.0
[0.9.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.8.0...0.9.0
[0.8.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.1.0...0.1.0

