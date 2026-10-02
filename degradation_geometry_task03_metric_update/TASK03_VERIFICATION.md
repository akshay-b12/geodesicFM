# Task 3 update verification

## Scope

The incremental ZIP adds exactly these nine files beneath the existing `code_setup` directory:

```
configs/metric.yaml
geometry_lab/metric.py
geometry_lab/metric_grids.py
geometry_lab/estimate_metric.py
geometry_lab/metric_analysis.py
geometry_lab/metric_plots.py
tests/test_metric.py
TASK03_README.md
TASK03_VERIFICATION.md
```

No earlier files, datasets, manifests or configurations are replaced. The existing Task 2 modules provide the operators, AD Jacobians and analytic derivative references. Do not extract this ZIP inside `code_setup`, which would create a nested folder. Extract into its parent instead.

## Verified environment

Python 3.12.14, PyTorch 2.7.1+cpu, NumPy 2.3.5 and Matplotlib 3.10.8 on Linux CPU. The supplied commands are also suitable for the user's Windows project directory. Windows and CUDA were not available for execution here; CPU is the default. No new Python dependency is introduced beyond Task 2.

## Checks performed

The complete unittest suite passed: **62 tests**, comprising the 44 existing tests plus 18 Task 3 tests. Task 3 coverage includes:

- Correct RGB-scalar normalization and off-diagonal inner products.
- Averaging realization-level Grams before averaging randomness.
- Equal image weights even when the general averaging helper receives unequal crop counts.
- Correct fixed coordinate scaling, explicit rank thresholds and preservation of singular matrices.
- Reporting tolerated negative eigenvalues without modifying them; rejection of substantial negative eigenvalues, invalid scales, asymmetry and nonfinite inputs.
- Affine closed forms, constant-image rank deficiency and exact orientation null directions at isotropy.
- Deterministic common noise fields and amplitude-independent additive-noise metrics.
- Agreement of the production estimator with averaged analytic Gram references for all 11 registered families, including both noise and haze composition orders.
- Correct axial grid neighbors and scaled parameter steps; invalid bounds and duplicate points rejected.
- Resume identity checks, checkpoint content checksums, partial-write recovery and changed-source rejection.
- An end-to-end fixture with separate training and validation means, all prepared crops, missing-checkpoint rejection and recovery.
- The fact that the averaged rank can exceed the rank of individual images.

The standalone synthetic controls passed **207 cases with zero failures**. They cover three patterns (textured, constant, impulse), all configured default grid points for the initial four families, and Task 2 registry points for the other seven families. Every case compares the AD Gram matrix with the closed-form/chain-rule reference. Additional gates check affine moments, zero blur sensitivity on constant images, the zero orientation direction at equal widths, and zero airlight sensitivity at zero haze beta. The largest absolute Gram-entry difference was **1.77e-16**, within the per-case acceptance threshold `1e-12 + 1e-10 * max(abs(reference))`.

An end-to-end synthetic pilot completed on two training images and one validation image, one crop each, over all 47 default parameter points. A synthetic full run completed on those same three images with both prepared crops each. The full run computed **354 Jacobians / 558 directional JVPs** and produced 12 committed image/operator checkpoints, eight operator/split summaries, eight overview plots and four two-dimensional grid plots, each plot in PNG and PDF. Full-grid rank maps were visually inspected and use discrete integer rank color scales. Resume reused all completed checkpoints, and aggregation/plotting can be rebuilt without source-image access or differentiation.

On this synthetic full dataset, the raw affine and additive-noise metrics were exactly constant across their respective parameter grids. The raw noise metric was approximately 0.999308 on training and 1.000569 on validation, consistent with a finite-sample squared-noise statistic. The blur metrics varied across their sampled widths. The default synthetic averaged metrics had ranks 2, 1, 1 and 2 for intensity, noise, isotropic blur and fixed-angle anisotropic blur respectively. These are implementation checks; they are not results on the user's DIV2K dataset and do not establish statistical convergence.

## Reproduce

From the existing project folder:

```powershell
python -m unittest discover -s tests -v
python -m geometry_lab.estimate_metric --stage controls
python -m geometry_lab.estimate_metric --stage pilot --dry-run
python -m geometry_lab.estimate_metric --stage pilot
```

Use `--run-name` for a fresh experiment if an output directory already exists, or `--resume` for the exact same experiment. Full DIV2K estimation must be run locally against the user's prepared source data. See `TASK03_README.md` for the averaging formula, parameter grids, full-run commands, output schemas, rank interpretation and scope of the conclusions.
