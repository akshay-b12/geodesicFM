# Task 4: metric variation and eigenstructure

Task 4 reads Task 3's aggregate matrices and analyzes observable local directions, coordinate sensitivities, derivative correlations, spectral degeneracies, numerical-rank sensitivity and metric variation. It uses NumPy, PyYAML and Matplotlib; it does not load images, recompute Jacobians or import PyTorch. It can run while Task 3 continues on other data.

## Install and run

Extract `degradation_geometry_task04_analysis_update.zip` into `E:\SRIB\cvpr_2027\geodesic_flow_matching`, the parent of the existing `code_setup` folder. All added Python files live in a new `task04_analysis` package **outside** `geometry_lab`. Task 3 hashes the `.py` files under `geometry_lab`; this update adds no files there and changes none of the existing modules, configurations, manifests or outputs. Installing Task 4 therefore leaves Task 3's code fingerprint unchanged. The new configuration is `configs/task04.yaml`.

Use the current Task 2/3 environment. To run from PowerShell:

```powershell
Set-Location E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup
python -m unittest discover -s tests -p test_task04.py -v
python -m task04_analysis --run outputs/task03/pilot --controls-run outputs/task03/controls --output outputs/task04/pilot
```

Use your actual run-directory names if they differ. `--controls-run` is optional. The main `--run` must contain image-metric aggregates from a completed pilot/full/synthetic image run; the controls stage is supplied separately because it stores matrices in `controls.json`.

Other examples:

```powershell
python -m task04_analysis --run outputs/task03/pilot --output outputs/task04/pilot_no_controls
python -m task04_analysis --run outputs/task03/full --operators isotropic_blur --output outputs/task04/isotropic_full
python -m task04_analysis --run outputs/task03/pilot --splits train --output outputs/task04/pilot_train
python -m task04_analysis --run outputs/task03/pilot --output outputs/task04/pilot_tables --no-plots
```

By default, all operator/split aggregates declared in the source run are required. Explicit `--operators` and `--splits` restrict the requested analysis; each requested aggregate must still cover all images selected for that split in the run metadata. The existing Task 3 estimator writes aggregate files after all requested checkpoints finish, so an interrupted run may have checkpoints but no usable aggregate. Task 4 does not silently create an estimate from that prefix. The completed pilot is sufficient for implementation and exploratory analysis while the full job runs.

Choose a new output folder for a new analysis. `--refresh` rebuilds an existing Task 4 destination only when the identity still matches: same input file hashes, settings, code and NumPy/Python runtime. A changed input or configuration needs a new output folder. Outputs inside the Task 3 source tree or controls tree are refused. Task 4 reads the original matrices without modifying them.

## Files to share for analysis

For the **pilot**, share a ZIP preserving these relative paths:

| File | Required? | Purpose |
|---|---|---|
| `pilot/run.json` | Yes | Coordinates, scales, grid, selected images/crops, protocol and identity |
| `pilot/analysis/<operator>/train/metric.npz` | Yes for training analysis | Mean and per-image training matrices |
| `pilot/analysis/<operator>/valid/metric.npz` | Yes for validation/split comparison | Mean and per-image validation matrices |
| `pilot/summary.json` | Recommended | Cross-check against the Task 3 coverage/results |

With the default configuration, include eight `metric.npz` files: training and validation for `affine_intensity`, `gaussian_noise`, `isotropic_blur` and `anisotropic_fixed_theta`. Include any additional families you enabled. Preserve folders rather than renaming the identically named NPZ files.

For the **Task 3 synthetic controls**, share:

- `controls/run.json`
- `controls/controls.json`

Those two JSON files include the control matrices, scales, parameter points and passed analytic checks. This is the Task 3 controls output, not the earlier Task 2 Jacobian control summaries.

If you also ran a **synthetic image dataset** through Task 3's pilot/full estimator, share that run's `run.json` and its `analysis/<operator>/<split>/metric.npz` files in the same format as the pilot. Its `summary.json` is recommended. Synthetic image-run aggregates and synthetic controls are different kinds of input; neither is a substitute for the DIV2K pilot.

The NPZ inputs contain `points`, `scales`, `mean_metric`, `per_image_metric`, `image_ids` and `crop_counts`. Later-added Task 3 arrays such as `per_image_scaled_rank` are optional because Task 4 recalculates its diagnostics from the matrices. Imports use `allow_pickle=False` and require float64 metrics. They verify shapes, points/scales, image order/counts and the equal-image mean. Input files and source metadata are hashed into the Task 4 analysis identity. This records the supplied files; it is not a cryptographic proof that the aggregate was independently reconstructed from every original checkpoint.

## Mathematical conventions

Task 3 estimates the mean of realization-level Gram matrices, with equal scalar weighting within the measured RGB core, realizations within each crop, crops within each image and images within each split. Task 4 preserves those matrices and that interpretation. Training and validation are analyzed separately.

The raw metric uses the registered physical coordinates. For fixed characteristic scales `S = diag(scales)` and `xi = S z`, the scaled metric is `S G S`. The scales are supplied by Task 3 and are not fitted from images. This makes the reporting units explicit; eigenvalues, conditions, weak-vector components and thresholded ranks remain coordinate dependent.

For a local physical perturbation `v`, `v.T @ G @ v` measures the expected squared RMS image change to first order. The diagonal square roots give per-coordinate RMS sensitivity. Off-diagonal correlations describe alignment between two derivative images and are undefined if either derivative norm is zero.

### Spectra, rank and weak subspaces

Eigenvalues are saved in ascending order. No jitter, matrix inversion or positive-definite replacement is used. Only the eigensolver input is symmetrized after the symmetry check. Tolerated roundoff-negative eigenvalues remain in the saved diagnostics and are counted. Nonfinite entries, substantial asymmetry or PSD violations raise an error.

Local numerical rank counts eigenvalues greater than `rank_atol + rank_rtol * max(lambda_max(point),0)`. Grid-wide-reference rank uses `rank_atol + rank_rtol * max_grid(lambda_max)` for all points in the same operator/split and coordinate system. The latter makes loss of absolute sensitivity relative to the sampled grid visible. Its reference depends on the analyzed grid; neither rule proves intrinsic dimension. Task 4's configurable thresholds are saved separately from the original Task 3 thresholds.

The default rank sensitivity study crosses three absolute tolerances (`1e-14,1e-12,1e-10`) with three relative tolerances (`1e-10,1e-8,1e-6`). A point changing rank across these choices is flagged. Agreement across this sweep is not statistical certainty.

Singular-at-threshold condition numbers are `null` in JSON and blank in CSV. A full-rank one-dimensional metric has condition one even when its magnitude is small; inspect absolute sensitivity as well.

The lowest eigenvalues are clustered using `eigengap_atol + eigengap_rtol * max(lambda_max,0)`. All eigenvalues within that distance of the lowest one form the reported weakest eigenspace. If its dimension exceeds one, Task 4 does not assign meaning to an arbitrary individual eigenvector. If it is one, the displayed vector's largest-magnitude component is made positive as a sign convention. A scaled vector is in `z` coordinates; multiplying by the characteristic scales gives its physical `xi` perturbation.

Neighbor/image/split weak-space comparisons use principal angles and projector distances. These are invariant to eigenvector signs and basis rotations within a repeated eigenspace. When compared weak-space dimensions differ, angles are explicitly unavailable. The cluster convention is distinct from the rank threshold: the weakest eigenspace is not necessarily the numerical nullspace. Its reported dimension is not intrinsic dimension.

Eigenvalue fractions and participation ratio describe spectral concentration. They use nonnegative contributions for that calculation, leaving the stored eigenvalues untouched. Participation ratio can differ from thresholded rank and does not infer manifold dimension. Exactly zero matrices have no fractions or participation ratio.

### Variation, image spread and split comparison

Neighbors differ in exactly one coordinate and are adjacent among the sampled values with their other coordinates held fixed. Irregular point lists get only existing axial pairs. Differences use the Frobenius norm of scaled matrices, and steps are divided by the corresponding characteristic coordinate scale. Relative differences divide by the larger endpoint norm; zero-to-zero change is zero.

The reference is the **actual sampled point** closest to the coordinatewise median in characteristic units. No matrix is interpolated at an unsampled median. Reference-relative variation and neighbor variation describe a fixed coordinate chart; neither is curvature.

Per-image descriptions include rank range, full-rank fraction, relative deviations from the mean, entrywise sample standard deviations and weakest-space agreement where the dimensions permit it. Standard deviation is unavailable with one image; these quantities are not standard errors or confidence intervals. The mean metric can have higher rank than each image's matrix because different images can reveal different directions. These descriptions prepare for Task 5's bootstrap/content-stability analysis without performing it.

Train/validation comparisons use corresponding points and the same fixed scales. Metrics are not pooled. Rank/weak-space differences remain descriptive and have no significance claim.

## Output files

| Output | Content |
|---|---|
| `analysis_run.json` | Task 4 settings, input hashes, source identity, code hashes and runtime |
| `summary.json` | Coverage, rank ranges, threshold-sensitive points, degeneracies and split comparisons |
| `REPORT.md` | Human-readable findings and interpretation limits |
| `<operator>/<split>/field_analysis.json` | Full raw/scaled spectra, correlations, weak bases, threshold sweep, neighbor changes and image descriptions |
| `<operator>/<split>/point_analysis.csv` | Flat coordinate, sensitivity, eigenvalue, rank, correlation and weak-direction results |
| `<operator>/<split>/rank_sensitivity.csv` | Numerical rank for each tolerance pair and point |
| `<operator>/<split>/neighbor_analysis.csv` | Axial metric and weak-space changes; exists when edges exist |
| `<operator>/split_comparison.csv` | Corresponding training/validation point comparisons |
| `controls_analysis.json` | Separate operator/pattern spectral and variation analyses of supplied controls |

Figures are exported as PNG and PDF: `spectra_and_rank`, `rank_sensitivity`, `neighbor_variation` where edges exist, and `identifiability_grid` for nontrivial 2D Cartesian grids. Log spectrum plots explicitly count omitted nonpositive eigenvalues, which remain available in JSON/CSV. Rank color scales are discrete. Parameter-grid cells represent sampled points rather than a fitted continuous metric field. Spectral curves are sorted eigenvalues, not tracked physical eigenvector branches through crossings.

`ANALYSIS_COMPLETE` means the requested imports and calculations completed. It is not a geometry acceptance decision. Synthetic outputs validate arithmetic; pilot outputs support exploratory observations. Full-data stability, coordinate-invariance validation, curvature and geodesic checks remain separate subsequent tasks.
