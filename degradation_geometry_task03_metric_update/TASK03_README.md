# Task 3: image-averaged degradation metrics

This update uses the float64 automatic-differentiation Jacobians verified in Task 2. It estimates inner products, local observable directions and descriptive changes with degradation parameters. It retains singular metrics and prepares image-level data for later uncertainty analysis. It does not fit geodesics, estimate curvature, bootstrap confidence intervals or select an intrinsic dimension.

## Installation and first run

Extract `degradation_geometry_task03_metric_update.zip` into `E:\SRIB\cvpr_2027\geodesic_flow_matching`, the **parent** of your existing `code_setup` folder. The ZIP adds the files listed in its verification document. It contains no existing configuration, degradation implementation, manifest or dataset. Keep your configured `configs/default.yaml` and prepared train/validation manifests.

The existing Task 2 environment is sufficient: PyTorch, NumPy, Pillow, PyYAML and Matplotlib. If needed, install the existing requirements:

```powershell
Set-Location E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup
python -m pip install -r requirements_task02.txt
python -m unittest discover -s tests -v
python -m geometry_lab.estimate_metric --stage controls
python -m geometry_lab.estimate_metric --stage pilot --dry-run
python -m geometry_lab.estimate_metric --stage pilot
```

The controls use synthetic data and check all 11 verified families. No real images are read in that stage. The pilot uses deterministic, evenly spaced image selections: up to four training images and two validation images, one prepared crop per image. For the planned DIV2K split, these are train 0001, 0267, 0533, 0800 and validation 0801, 0900, assuming the original sorted manifests. The pilot uses the full configured parameter grids; it reduces data coverage, not parameter coverage.

Read `outputs/task03/controls/controls.json` and `outputs/task03/pilot/summary.json`, then inspect the point summaries and plots. A successful pilot verifies execution and arithmetic on a small sample; it is not a converged DIV2K estimate.

Run the complete prepared data after reviewing the pilot:

```powershell
python -m geometry_lab.estimate_metric --stage full --dry-run
python -m geometry_lab.estimate_metric --stage full
```

`--dry-run` verifies configuration, selected source hashes and manifest consistency, and prints the exact work count without creating a run directory. The default full experiment with 800 training images, 100 validation images and four crops each requires **212,400 Jacobian evaluations / 334,800 directional JVPs**. The default six-image pilot requires 354 / 558. Wall time depends strongly on crop size, CPU and operator; use the pilot to assess it. Images are decoded once per image during estimation, and Jacobians are reduced immediately rather than stored.

An interrupted run resumes with the same arguments plus `--resume`:

```powershell
python -m geometry_lab.estimate_metric --stage full --resume
```

Resume reuses completed **image/operator** checkpoints. An interrupted image/operator is recomputed; completed images remain usable. The run identity includes grids, sampling, source/manifest hashes, code hashes, runtime, float64 settings, seeds and selected records. Changed inputs require a new `--run-name`; silently mixing experiments is refused. A damaged committed checkpoint is rejected with its path. Remove that checkpoint's `.npz` and `.json` marker together and resume to recompute it. An uncommitted `.npz` is recomputed automatically. Use one writing process per run directory.

You can process a single family or split using a distinct run name:

```powershell
python -m geometry_lab.estimate_metric --stage full --operators isotropic_blur --run-name isotropic_full
python -m geometry_lab.estimate_metric --stage full --operators anisotropic_fixed_theta --run-name anisotropic_full
python -m geometry_lab.estimate_metric --stage full --split train --run-name train_full
```

Both splits are estimated separately by default. The training metric is the geometry estimate; validation provides a separate descriptive comparison. They are never pooled. No dataset resizing, padded clean crops, output clipping or new crop sampling is performed.

## Definition and averaging

For an image crop with `M = C H W` measured output scalars, let `J` have shape `(k,C,H,W)`, with one derivative image per parameter. The crop/realization metric is

\[
G_{ij}^{(n,c,r)}(\xi)=\frac{1}{CHW}\sum_{a,h,w}
\frac{\partial D(x_{n,c},\xi;\epsilon_{n,c,r})_{a,h,w}}{\partial\xi_i}
\frac{\partial D(x_{n,c},\xi;\epsilon_{n,c,r})_{a,h,w}}{\partial\xi_j}.
\]

The estimator first averages realizations within a crop, then crops within an image, then gives every image equal weight:

\[
\widehat G(\xi)=\frac1N\sum_{n=1}^N
\frac1{C_n}\sum_{c=1}^{C_n}\frac1{R}\sum_{r=1}^R G^{(n,c,r)}(\xi).
\]

Deterministic operators use one realization. Stochastic operators use four by default. This is `E[J J^T]` in the parameter-first layout, **not** the Gram matrix of `E[J]`. The distinction is essential: symmetric random noise derivatives could cancel in `E[J]`, despite having positive squared sensitivity.

Gaussian fields are fixed over all parameter points of a crop/realization. Their stable seeds depend on the configured seed, image ID, crop index and realization index; they omit the parameter point and operator name. This makes parameter changes paired and shares compatible fields between composition orders. Different crops/images have separate seeds. Unit depth and the synthetic horizontal depth ramp remain the Task 2 definitions. Haze experiments describe those stipulated fields, not measured DIV2K scene depth.

Operators act on the original physical context, and only the core contributes to the metric. All RGB channels are included with equal scalar weight. Image values are in the prepared color space (`srgb` by default); changing to `linear_rgb` changes the scientific metric and requires consistent preparation. Noise and affine intensity are unclipped.

Locally, for a parameter perturbation `v`,

\[
v^T\widehat G(\xi)v \approx
\mathbb E[\|D(x,\xi+v;\epsilon)-D(x,\xi;\epsilon)\|^2/(CHW)].
\]

This describes image-averaged **local** squared change under paired auxiliary fields. It is not a likelihood Fisher information matrix or a proof of global parameter uniqueness.

## Default grids and optional families

`configs/metric.yaml` supplements `default.yaml`; it does not override it. Cartesian points use Task 2 parameter order, with the last coordinate varying fastest.

| Family | Grid | Points | Characteristic scales |
|---|---|---:|---|
| Affine intensity | `a = [0.5,1,1.5]`, `b = [-0.1,0,0.1]` | 9 | `(1,0.1)` |
| Gaussian noise | `sigma_noise = [0,0.05,0.1,0.2]` | 4 | `(0.1)` |
| Isotropic blur | `sigma = [0.5,0.75,1,1.25,1.5,2,2.5,3,3.5]` | 9 | `(1)` |
| Anisotropic blur at fixed angle | Both widths `[0.5,1,2,3,3.5]`; angle `0.31` radians | 25 | `(1,1)` |

The width units are pixels, angle units radians, and intensity/noise values use normalized image units. Kernel support remains 33 and halo at least 16 under the verified setup. The differentiable blur chart starts at positive sigma; sigma zero is not introduced as a blur anchor. Zero additive-noise amplitude is supported and still has a nonzero pathwise derivative.

Commented examples enable full anisotropic orientation, unit/toy haze and both composition orders. Use either `axes` with exactly the registered coordinate names or an explicit `points` list. Coordinates must be finite, distinct and within the Task 2 bounds. Axes must be strictly increasing. Optional explicit point sets are not interpolated; neighbor analysis only uses pairs that actually differ in one coordinate. Controls check registry points for families absent from the configured grids.

## Reading the outputs

Each run has:

| File | Content |
|---|---|
| `run.json` | Complete protocol, selected records, hashes and run identity |
| `checkpoints/<operator>/<split>/*.npz` | One image's mean, all crop means, crop IDs and realization trace spread |
| Matching checkpoint `.json` | Commit marker, image ID, run identity and content checksum |
| `summary.json` | Coverage, rank ranges, metric norm ranges and separate split comparisons |
| `analysis/<operator>/<split>/metric.npz` | Points, scales, mean matrix, per-image matrices, image IDs, crop counts, per-image scaled ranks and mean realization trace spread |
| `point_summary.csv` | Parameter values, raw entries, image spread, raw/scaled eigenvalues, ranks, condition and scaled weak direction |
| `diagnostics.json` | Full eigenvectors, correlations, thresholds, zero-direction handling and diagnostic details |
| `neighbor_variation.json/.csv` | Adjacent axial point changes (CSV only when edges exist) |
| `metric_overview.png/.pdf` | Raw entries, scaled eigenvalues and scaled numerical rank |
| `metric_grid.png/.pdf` | Two-dimensional Cartesian maps, including singular condition regions |

Load numeric outputs without pickle:

```python
import numpy as np
d = np.load("outputs/task03/full/analysis/isotropic_blur/train/metric.npz", allow_pickle=False)
points = d["points"]                         # (P,k)
G = d["mean_metric"]                        # (P,k,k), raw physical coordinates
image_G = d["per_image_metric"]              # (N,P,k,k), image-level sampling units
S = d["scales"]
G_scaled = G*S[None,:,None]*S[None,None,:]
```

Rebuild summaries/plots from committed matrices without differentiating or loading source images:

```powershell
python -m geometry_lab.metric_analysis --run outputs/task03/full
python -m geometry_lab.metric_analysis --run outputs/task03/full --no-plots
```

This rebuild validates all required checkpoint identities and checksums, and refuses incomplete runs. It uses the diagnostic thresholds saved in the run. The per-image matrices are retained for later image-level bootstrap; do not bootstrap the crops as independent images.

## Inner products, identifiable directions and variation

**Entries:** `G_ii` is the squared RMS sensitivity to coordinate `i`. `sqrt(G_ii)` gives the RMS change per unit coordinate perturbation. `G_ij` is the inner product between two derivative images. Positive or negative off-diagonal values describe aligned or opposed local changes. Correlation divides by the two derivative norms and is undefined for a zero direction; those entries are JSON `null`.

**Scaling:** If `xi = S z` with the fixed positive diagonal scales in the registry, the scaled metric is `S G S`. Raw entries remain the physical-coordinate metric. Scaled diagnostics compare perturbations in the stipulated reference units. These are fixed scales, not image-fitted whitening or unit-diagonal normalization. Eigenvalues, condition numbers, numerical ranks and vector components can change with coordinate scaling. They are not coordinate-invariant geometric conclusions.

**Eigenvalues:** Small eigenvalues mark combinations that make little local image change. Eigenvectors are columns, ordered by ascending eigenvalue. `scaled_weak_vector_*` is the lowest-eigenvalue unit vector in `z` coordinates; its physical perturbation is `S` times that vector. Its sign is arbitrary. Near repeated eigenvalues, individual vectors are unstable and only the corresponding subspace is meaningful.

**Rank:** An eigenvalue counts as active when it exceeds `rank_atol + rank_rtol * max(lambda_max,0)` in the stated raw or scaled coordinates. Defaults are `1e-12` and `1e-8`. This is a numerical reporting convention. A weak nonzero direction below it is not proven mathematically unidentifiable. Singular-at-threshold conditions are JSON `null` and blank CSV cells, not fabricated finite condition numbers. Exactly zero matrices have rank zero.

No jitter, inversion, eigenvalue clipping or positive-definite replacement is applied to saved metrics. Small negative eigenvalues permitted by the explicit PSD roundoff tolerance remain stored and are counted. Nonfinite values, asymmetry and negative eigenvalues beyond that tolerance stop the run. The RMS display alone maps a tolerated negative diagonal to zero; it does not modify the matrix or its stored eigenvalues.

A full-rank one-coordinate metric has condition number one regardless of its magnitude. Read its eigenvalue or direction RMS to assess absolute sensitivity; condition alone cannot identify a weak one-dimensional signal.

**Image coverage:** The averaged metric's nullspace is the intersection of per-image nullspaces in the exact PSD case. Different images can reveal different directions, so an averaged matrix may have larger rank than each image's matrix. The per-image rank range and full-rank fraction expose this distinction. They do not describe per-crop rank. Numerical classification can also differ because thresholds are applied separately.

**Variation:** Neighbor comparisons use `S G S`, the Frobenius norm of its difference and the parameter step in characteristic units. Relative change divides by the larger endpoint norm; zero-to-zero change is zero. These quantities describe how the metric varies within this chart. A one-dimensional metric can vary strongly while having no intrinsic Riemann curvature. Train/validation differences use the same symmetric relative normalization and remain descriptive.

`image_std_G_*` is the sample standard deviation across image matrices, not a standard error or confidence interval; it is blank when only one image is present. `mean_realization_trace_std` is the image/crop-averaged sample standard deviation of realization-level **raw traces**, not Monte Carlo error of the final metric. Deterministic operators store zero for that spread. Image/bootstrap uncertainty, crop convergence and realization convergence remain future analyses.

## Expected controls and scientific limits

For affine intensity, `J_a = x`, `J_b = 1`, so the metric is `[[mean(x²),mean(x)],[mean(x),1]]`. Its determinant is the pooled scalar variance. On a constant image it is rank one: changes in gain and offset can compensate. Its metric is independent of `a,b` in the unclipped model.

For Gaussian noise, `J_sigma = epsilon`, so the sample metric is `mean(epsilon²)`. It is independent of noise amplitude under fixed fields, including at zero. The population expectation is one; a finite sample need not equal one exactly.

Normalized blur preserves constant images, so their width derivatives and Gram matrices vanish up to arithmetic roundoff. Textured-image blur sensitivities vary with sigma; monotonicity or a universal rank is not assumed. With equal anisotropic widths, the **orientation** derivative is exactly zero. The two width directions can remain distinguishable in the fixed-angle chart; equal widths do not force that two-coordinate metric to rank one. For haze at beta zero, the airlight derivative vanishes. Composition order changes the derivatives and can change the metric.

Controls compare AD-based Gram matrices with the existing closed-form/chain-rule Jacobians for every verified family and additionally check these exact zero and affine identities. A `COMPLETE` data summary means every requested checkpoint was aggregated and passed finite/symmetry/PSD checks. It does not certify all directions identifiable, statistical convergence, global injectivity or nonsingular charts throughout a continuous domain.
