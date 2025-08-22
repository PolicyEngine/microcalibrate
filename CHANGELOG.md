# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
