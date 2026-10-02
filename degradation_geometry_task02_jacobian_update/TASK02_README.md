# Task 2 — Jacobian estimation and verification

The earlier roadmap assigns **Task 2 to Jacobian verification**, and Task 3 to
operator-induced metric estimation. This update implements the Jacobian step.
Use your already prepared DIV2K manifests. Do not regenerate the crops or change
your existing dataset paths just to install this update.

## Install this incremental update

The archive contains only new files, inside a top-level `code_setup` directory.
Extract it into:

```text
E:\SRIB\cvpr_2027\geodesic_flow_matching\
```

That adds these files to your existing `code_setup`:

| File | Purpose |
|---|---|
| `geometry_lab/jacobian.py` | Reusable forward-AD and finite-difference Jacobian estimators; error gates and reverse checks |
| `geometry_lab/jacobian_cases.py` | Domains, scales, parameter points, fixed auxiliary fields, analytic references and composition chain rules |
| `geometry_lab/validate_jacobian.py` | Runs the four validation stages and writes auditable reports |
| `geometry_lab/jacobian_plots.py` | Error-versus-step plots and example derivative maps |
| `configs/jacobian.yaml` | Validation settings, sample counts and explicit acceptance thresholds |
| `tests/test_jacobian.py` | Mathematical tests, integration checks and deliberately faulty negative controls |
| `requirements_task02.txt` | Matplotlib dependency for exported plots |
| `TASK02_README.md` | This explanation and execution instructions |
| `TASK02_VERIFICATION.md` | Verification results from the development environment |

The update does not contain `configs/default.yaml`, your original operator files,
or Task 0 outputs. Your configured dataset paths and existing manifests remain
in place. All commands below run from your project root:

```powershell
Set-Location 'E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup'
.\.venv\Scripts\python.exe -m pip install -r requirements_task02.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If you use another environment, substitute its Python executable. Existing
PyTorch installation suffices; development verification used PyTorch 2.7.1 CPU.
You do not need to reinstall the original package. The new modules are available
from the project directory or your existing editable installation.

## What exactly is differentiated?

For a fixed clean context crop, fixed noise/depth fields and fixed kernel support,
define

\[
F_x(\xi)=\operatorname{core\_crop}(D_\xi(x_{\rm context})).
\]

The estimator returns

\[
J_i(x,\xi)=\frac{\partial F_x(\xi)}{\partial\xi_i}.
\]

**Tensor layout is parameter first:** `(k, C, H, W)`. For your 256 × 256 RGB
cores and a three-parameter operator, this is `(3, 3, 256, 256)`. `J[i]` is an
entire derivative image for coordinate `i`. The image, noise realization, depth,
core coordinates, composition order, kernel support and color representation are
held fixed while differentiating. No clipping, PNG conversion or random resampling
occurs inside the differentiated mapping.

We use one `torch.func.jvp` per coordinate, with the standard basis tangent `e_i`:

\[
J_i=J_F(\xi)e_i.
\]

This costs one forward-AD pass per parameter rather than a reverse pass per image
pixel. The code does not construct a massive image-pixel-by-image-pixel Jacobian.
The forward-AD estimator is the proposed production estimator for Task 3; finite
differences and independent references verify it.

## Four stages, in order

### 1. Synthetic controls

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.validate_jacobian --stage controls
```

This uses textured, constant and impulse images, with no DIV2K loading. The
default run has eleven operator families, multiple parameter points, two fixed
noise realizations for stochastic families and ten step sizes. Each individual
case must pass; an average cannot conceal one failed coordinate.

References include:

- Affine intensity: `dD/da=x`, `dD/db=1`.
- Additive noise: `dD/dsigma=epsilon`, including amplitude zero.
- Haze: `dD/dbeta=depth*t*(A-x)` and `dD/dA=1-t`, including `beta=0`, where the
  airlight derivative must vanish.
- Isotropic and anisotropic sampled Gaussian kernels: closed-form derivatives
  including the derivative of kernel normalization.
- Blur/noise and blur/haze compositions: explicit chain-rule references for both
  orders, with all auxiliary fields frozen.
- Equal-width anisotropic blur: orientation derivative must vanish; the tests
  additionally check that the two width derivatives sum to the isotropic width
  derivative.

Start the DIV2K stage only after reading a `PASS` control summary.

### 2. DIV2K pilot, float64

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.validate_jacobian --stage pilot
```

This reuses your Task 0 manifests and verifies source-file hashes. The defaults
select four training images and two validation images, spread deterministically
across their sorted indices, and the first saved crop from each. It uses **the
original prepared crop size**, normally 256 × 256; it does not silently shrink
or resample your DIV2K crops.

This numerical verification does not require running all 900 images. It checks
the derivative implementation on independent natural-image contents. Later metric
estimation uses the full planned sampling protocol. Increase the pilot counts in
`configs/jacobian.yaml` for broader confirmation after the initial run. The
validation subset is a correctness check, not data for fitting a metric.

The float64 gate combines:

1. Full derivative-image finite-difference sweeps for every coordinate.
2. Closed-form/chain-rule references for every registered family.
3. Three independent reverse-mode random output projections per case.
4. Exact repeated-forward determinism for the frozen operator.
5. Explicit degeneracy checks where a derivative should be zero.

### 3. Float32 sensitivity

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.validate_jacobian --stage precision
```

The default uses one training crop and compares float32 AD Jacobians with float64
AD Jacobians. It also runs the finite-difference and reference checks in float32.
The original image, noise and depth fields are produced in float64 and cast to
float32 for this comparison. Noise is **not regenerated using the float32 RNG**,
which could produce a different realization even with the same seed.

Float32 differences at small steps can be dominated by cancellation. Consequently,
`REVIEW_REQUIRED` here need not invalidate a passing float64 implementation.
Inspect whether the failure comes from the float32 finite-difference sweep, its
analytic/reverse checks, or the AD32-versus-AD64 comparison. Keep float64 as the
baseline for metric/curvature work unless this study justifies a faster dtype.

The full pilot can take substantial CPU time, especially for 2-D anisotropic
convolution. Start with controls; use the focused blur command below to review
the main operator before the full pilot if useful. The console reports progress
by patch and operator. Keep CPU float64 as the initial reference; CUDA can be
selected in the settings afterward, with deterministic operations and TF32
disabled. An unsupported deterministic/AD operation is recorded as a failed case,
not silently replaced by a different backend.

### 4. Fixed-kernel-support sensitivity

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.validate_jacobian --stage support
```

This compares the original 33 × 33 kernel with a 49 × 49 kernel using the
same measured core, original-image context, image values, noise and depth for
both evaluations. The default selects two training images and one validation
image, and only blur-containing families.

The 49 × 49 kernel needs a 24-pixel physical halo. The runner rereads the original
HR image around the saved core coordinates to obtain that context; it does not
enlarge the measured core or substitute reflection padding for missing original
pixels. If the source image cannot supply the larger halo, it records that entry
and marks the study `INCOMPLETE`. Inspect `unavailable_support.json`; select an
eligible subset explicitly if needed, using the option described below. No need
to overwrite your Task 0 manifests.

A support discrepancy measures sensitivity of the chosen finite-kernel operator.
It is separate from AD correctness: AD and finite differences can agree perfectly
for a support approximation whose derivatives change under a larger kernel.
If support requires revision, revisit the operator definition and physical halo
before metric estimation; do not silently mix Jacobians from different kernels.

## Step sizes and boundary handling

For a dimensionless epsilon, coordinate `i` uses

\[
h_i=\epsilon s_i,
\]

where `s_i` is a fixed characteristic unit, recorded in the operator registry:

| Coordinate | Fixed scale | Units |
|---|---|---|
| Blur width | 1 | Original-image pixels |
| Blur angle | 1 | Radians |
| Noise amplitude | 0.1 | Image-intensity units |
| Affine gain | 1 | Gain |
| Affine offset | 0.1 | Image-intensity units |
| Haze beta and airlight | 1 each | Units for the fixed synthetic depth/intensity convention |

Scales are not adapted to image content or the current parameter value. The CSV
records both epsilon and the actual absolute step.

For an interior point:

\[
J_i^{\rm FD}(h_i)=\frac{F(\xi+h_ie_i)-F(\xi-h_ie_i)}{2h_i}.
\]

At a lower boundary, a second-order forward stencil is used:

\[
J_i^{\rm FD}(h_i)=\frac{-3F(\xi)+4F(\xi+h_ie_i)-F(\xi+2h_ie_i)}{2h_i}.
\]

The analogous second-order backward stencil is used at an upper boundary.
Every row labels its scheme as `central2`, `forward2`, `backward2`, or
`unavailable`. A step is never silently shrunk to fit the domain. An unavailable
stencil is represented by a failed/nonfinite diagnostic, not a zero derivative.
The reusable API can instead skip boundary stencils with `boundary="skip"`.

This avoids evaluating negative physical noise amplitudes or blur widths outside
the declared study domain merely to obtain a centered difference. These are
derivatives of the smooth operator restricted to the chosen chart, with a
one-sided numerical approximation at its endpoints.

## Acceptance rule and near-zero directions

For each coordinate, the numerical gate is

\[
\operatorname{RMS}(J_i^{\rm FD}-J_i^{\rm AD})
\le a_{\rm tol}+r_{\rm tol}\operatorname{RMS}(J_i^{\rm AD}).
\]

Default float64 FD tolerances are `atol=1e-8`, `rtol=1e-5`. Analytic comparisons
and reverse projections use separate, tighter explicit thresholds. See the YAML
for all values. These are declared numerical acceptance rules, not thresholds
for proving useful geometry or intrinsic dimensionality.

The reference is classified near zero below `signal_floor=1e-10`. Relative error
and cosine are then written as null rather than dividing by an artificial floor.
The absolute RMS gate remains active. This matters for constant images, equal-axis
orientation and identity-haze airlight directions. A tiny derivative is not a
reason to declare that an estimator failed or to invent a positive eigenvalue.

Each coordinate must have **at least two adjacent passing steps** in the sweep.
Its first qualifying epsilon interval is recorded. All step results remain in
the CSV, including the expected failures at overly large or overly small steps.
The software neither cherry-picks one best epsilon nor averages different
coordinates/cases into a passing score. A passing interval verifies this tested
derivative at the stated tolerance; a convergence slope remains an informative
plot diagnostic, and is not used as an automatic proof of correctness.

## Independent reverse-mode cross-check

For a normalized random output probe `w`, check

\[
\nabla_\xi\langle w,F(\xi)\rangle
=\big(\langle w,J_1\rangle,\ldots,\langle w,J_k\rangle\big).
\]

The left side uses `torch.autograd.grad` and the right side uses the assembled
forward-AD Jacobian. This catches indexing/scaling/layout errors and disagreement
between forward and reverse implementations. Random projections supplement the
full-image FD and analytic tests; they cannot replace them. The unit tests also
compare against a full reverse-mode Jacobian for a tiny known polynomial.

## Reports to inspect

Each stage writes a new folder such as `outputs/task02/pilot`:

| Output | How to use it |
|---|---|
| `summary.json` and `REPORT.md` | Read the overall status and coverage first |
| `case_summary.json` | Inspect individual failed cases and accepted epsilon intervals |
| `finite_difference_sweep.csv` | Per case/coordinate/step errors, schemes, cosine, signal strength and gates |
| `analytic_checks.csv` | Independent closed-form/chain-rule errors |
| `reverse_projection_checks.csv` | Per probe/coordinate forward-versus-reverse errors |
| `comparison_checks.csv` | Float32 or kernel-support comparisons, depending on the stage |
| `unavailable_support.json` | Source crops that cannot provide the larger physical context |
| `run_metadata.json` | Versions, source-code hashes, manifest hashes, settings, scales/domains and selected image indices |
| `plots/sweep_*.png` | Median and min-max errors over the listed cases; read the CSV for individual failures |
| `example_*.pt` | Selected raw Jacobians, context, noise/depth, parameters and FD epsilon |
| `plots/example_*.png` | Signed red-channel AD/FD derivative maps and residual maps |

Derivative maps use the same color limits for AD and FD. Residuals have their own
explicitly labeled color scale. Plots floor zeros at `1e-18` only for log display;
numeric gates use the unmodified values. Min-max bands are descriptive ranges,
not confidence intervals. Empty relative-error panels can occur when every
reference direction is near zero.

Statuses:

| Status | Meaning |
|---|---|
| `PASS` | All required checks for the documented cases passed |
| `FAIL` | A controls/pilot case failed; investigate before estimating a metric |
| `REVIEW_REQUIRED` | Precision or support diagnostic exceeded its tolerance |
| `INCOMPLETE` | Required operator coverage or larger-context entries are missing |

The CLI exits with code 0 only for `PASS`. A run directory is not reused or
overwritten. To rerun after a deliberate settings change:

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.validate_jacobian --stage pilot --run-name pilot_v2
```

For a quick focused run, for example:

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.validate_jacobian --stage pilot --operators isotropic_blur,anisotropic_blur --run-name blur_only
```

That report covers only the listed families. Run all planned families before
accepting the general estimator. `--no-plots` removes the Matplotlib requirement
and plotting cost while retaining CSV/JSON diagnostics.

## Explicit sample selection

Ordinary pilot selection spreads the specified image count across sorted indices.
If you need specific eligible support crops or an exact repeatable debugging set,
add stage-specific overrides to `configs/jacobian.yaml`:

```yaml
support:
  max_train_images: 2
  max_valid_images: 1
  crops_per_image: 1
  larger_kernel_size: 49
  train_image_indices: [0, 200]
  valid_image_indices: [30]
```

These are zero-based indices into the corresponding manifest `images` list, not
DIV2K filename numbers. The explicit lists override the corresponding maximum
counts. An empty list skips that split. Image indices and the number of available
saved crops are validated, and the chosen indices are recorded in provenance.
Use the saved core coordinates and original image dimensions to ensure a
24-pixel halo is available for a 49 × 49 kernel.

## API for your code review

Read the modules in the order `jacobian.py`, `jacobian_cases.py`, then
`validate_jacobian.py`. The low-level estimator can be used directly:

```python
import torch
from geometry_lab.datasets import ManifestDataset, core_crop
from geometry_lab.degradations import gaussian_blur
from geometry_lab.jacobian import jacobian_jvp, finite_difference_jacobian

item = ManifestDataset("outputs/task01/train_manifest.json")[0]
x = item["image"]
xi = torch.tensor([1.5], dtype=torch.float64)
fn = lambda p: core_crop(gaussian_blur(x, p, kernel_size=33),
                        item["halo"], item["crop_size"])

J_AD = jacobian_jvp(fn, xi)  # (1,3,256,256) with the standard protocol
FD = finite_difference_jacobian(fn, xi, steps=[1e-4], bounds=[(0.5,3.5)])
J_FD = FD.jacobian
```

For later derivatives of a metric with respect to parameters:

```python
xi = xi.clone().requires_grad_(True)
J = jacobian_jvp(fn, xi, create_graph=True)
```

The default detaches the Jacobian to avoid retaining unnecessary graphs during
verification/estimation. `create_graph=True` preserves parameter differentiation;
unit tests verify higher derivatives through a blur-Jacobian energy. That test
does not replace future numerical validation of metric derivatives and curvature.

For a normalized metric in the next task, parameter-first layout corresponds to
`A = J.reshape(k,-1)` and `A @ A.T / (C*H*W)`. This update deliberately stops before
dataset-level metric estimation.

## What to share before Task 3

Run the four stages, then provide:

1. Each stage's `summary.json`.
2. `case_summary.json` and the relevant CSVs if any case failed.
3. The isotropic and anisotropic sweep plots.
4. `comparison_checks.csv` for precision and support, and
   `unavailable_support.json` if support is incomplete.

These let us distinguish a coding issue, unsuitable finite-difference steps,
machine-precision effects, expected unidentifiable directions and a support
approximation that needs revision. Do not proceed to intrinsic curvature claims
on the basis of derivative agreement alone.

## Sources

- Forward AD: https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html
- Independent reverse gradients: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- Tiny reverse-Jacobian reference: https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
