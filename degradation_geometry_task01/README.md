# Degradation geometry — Tasks 0 and 1

This project implements the first two steps of the agreed pre-FLUX study:

1. **Task 0:** reproducible DIV2K sampling, fingerprints, crop manifests and basic content statistics.
2. **Task 1:** differentiable controlled degradation operators and progression previews.

There is no restoration training, metric estimation, curvature computation or claim-validation result yet. Those remain separate later tasks. Code comments explain the choices that affect the eventual induced metric.

## 1. Put this project in your chosen folder

Extract the archive so that these files appear directly inside:

```text
E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup\README.md
E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup\configs\default.yaml
E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup\geometry_lab\datasets.py
```

The archive has one top-level `code_setup` directory. Extract into its parent (`geodesic_flow_matching`) to obtain the paths above. If `code_setup` already exists and contains work, extract into a separate location first and copy reviewed files into it.

## 2. Set up Python on Windows

Open PowerShell. Use Python 3.10 or newer; the delivered code was verified on Python 3.12 with a CPU PyTorch build. This sequence does not require activating PowerShell scripts:

```powershell
Set-Location 'E:\SRIB\cvpr_2027\geodesic_flow_matching\code_setup'
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

If the Windows Python launcher `py` is unavailable, use the path to your installed Python executable in its place. An existing research environment can also be used. Installing PyTorch first lets you choose a CPU or CUDA build appropriate for that environment. Tasks 0 and 1 work on CPU; no GPU is required.

On Linux, use `python3 -m venv .venv`, then `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## 3. Set the actual DIV2K paths

Edit only these two fields in `configs/default.yaml` first:

```yaml
dataset:
  train_hr: E:/your_dataset_location/DIV2K_train_HR
  valid_hr: E:/your_dataset_location/DIV2K_valid_HR
  expected_train_images: 800
  expected_valid_images: 100
```

The download deliberately leaves both paths `null`, since your dataset location has not been provided. Use forward slashes in YAML Windows paths. Both directories should directly contain the original RGB HR PNGs, rather than another nested directory or LR images.

The program checks the specified image counts and fails if they do not match. For a deliberate small pilot, set those counts to the sizes of your pilot directories. Do not lower counts merely to conceal an incomplete full dataset.

No images are downloaded automatically. Do not merge the validation images into training. Here, “untouched validation” means not used to select the geometry or tune its estimator; Task 0 inventories it and fixes its crops, while Task 1 previews only training images.

## 4. Verify the installation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks use temporary synthetic images, so they run without DIV2K. They cover crop repeatability, source-file changes, RGB decoding, split leakage, real-image blur context, kernel identities, explicit composition order, and parameter gradients/JVPs.

For a complete CLI smoke test without DIV2K:

```powershell
.\.venv\Scripts\python.exe tools/make_demo.py
.\.venv\Scripts\python.exe -m geometry_lab.prepare --config configs/demo.yaml
.\.venv\Scripts\python.exe -m geometry_lab.preview --config configs/demo.yaml
```

These commands create a small synthetic dataset and previews in `outputs/demo`. They check the software pipeline only. Their images and results cannot support scientific conclusions about natural-image degradation geometry.

## 5. Run Task 0

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.prepare --config configs/default.yaml
```

Default sampling protocol:

| Setting | Value | Purpose |
|---|---|---|
| Training images | 800 | Main geometry estimation in later tasks |
| Validation images | 100 | Independent content confirmation in later tasks |
| Core crop | 256 × 256 | Region on which later derivatives/metrics are measured |
| Crops per image | 4 | Deterministic sampling of image content |
| Context halo | 16 pixels per side | Prevent artificial crop-edge blur effects |
| Returned context tensor | 3 × 288 × 288 | Operator input; crop back to the core afterward |
| Seed | 20260930 | Independent stable seed for each image/crop key |
| Intensities | RGB in [0,1], sRGB-coded | Explicit ambient image representation |
| Geometry dtype | float64 | Recommended for derivative-sensitive computations |

The program does not resize images, use augmentations, convert color channels silently, or substitute padding for missing crop context. Images too small for the protocol cause a clear error.

By default, 3,200 training crops and 400 validation crops are specified, without storing all crops as separate PNGs. Hashes identify the original image files. Split-leakage checks compare decoded pixels, catching duplicate images even if their PNG file encodings differ. Crops are generated from a seed keyed by `(global_seed, image_id, crop_index)`, so an unrelated image being added does not change existing crop coordinates. Random crops may overlap; they are not independent statistical units.

Expected outputs in `outputs/task01`:

| File | Contents |
|---|---|
| `train_images.txt`, `valid_images.txt` | Sorted source filenames |
| `train_manifest.json`, `valid_manifest.json` | Source directory, hashes, dimensions and complete crop records |
| `crop_coordinates.json` | Coordinates for both splits; top/left identify the core in the original image |
| `train_content_stats.csv`, `valid_content_stats.csv` | One row per image; averages of patch channel means, patch channel standard deviations and gradient energy |
| `dataset_summary.json` | Image/crop counts and image-weighted summaries |
| `config_resolved.yaml` | Config used by preparation, including any CLI path overrides |
| `provenance.json` | Software versions and manifest hashes |

Channel standard deviations in these statistics are **averages of within-patch standard deviations**, rather than the standard deviation of an entire image or dataset. Gradient energy is the mean squared adjacent-pixel difference, averaged over horizontal and vertical directions. These are descriptive checks, not a content-invariance test.

The image and crop manifests provide Task 3's grouping: first average patches within each image, then average images with equal weight. Later bootstrap whole `image_id` groups rather than individual crop records.

Running preparation into a nonempty output directory requires `--overwrite`. Prefer a new output directory for a changed sampling protocol. If you overwrite Task 0, regenerate Task 1 previews as well; old previews may correspond to an older manifest.

## 6. Run Task 1

```powershell
.\.venv\Scripts\python.exe -m geometry_lab.preview --config configs/default.yaml
```

This command selects training image index 0 and its crop index 0. Change `preview.image_index` and `preview.crop_index` to inspect another fixed training crop. Indices are zero-based. It verifies the source-file hashes and checks that the sampling config matches the saved manifest.

You receive eleven PNG grids under `outputs/task01/previews`, plus:

| File | Contents |
|---|---|
| `raw_tensors.pt` | Clean context/core, explicit noise/depth realizations and unclipped degraded core tensors |
| `preview_metadata.json` | Crop identity, parameters, operator names/order, dtype, seed, version and raw intensity ranges |

Every grid starts with the clean crop. PNGs clip intensity values solely for display. Always use the operators or `raw_tensors.pt` for numeric work, never the quantized/clipped previews.

For a rerun into the same preview folder, pass `--overwrite`.

## 7. Operator definitions and domains

All operators accept floating CHW or NCHW tensors and a shared parameter vector `xi`. A batch uses the same parameters for every image; per-image parameters can be applied in a loop. Parameter tensors retain their PyTorch autograd graph. The operators also work with `torch.func.jvp`, which will be useful for Task 2.

| Operator | Parameter vector | Definition / domain |
|---|---|---|
| `gaussian_blur` | `[sigma]` | Normalized sampled Gaussian; `sigma > 0`, in original-image pixels |
| `anisotropic_blur` | `[sigma_x, sigma_y, theta]` | Elliptical Gaussian; positive widths, angle in radians |
| `affine_intensity` | `[a,b]` | `a*x+b`, without clipping; mathematical control on real-valued parameters |
| `gaussian_noise` | `[sigma_noise]` | `x+sigma_noise*epsilon`; nonnegative amplitude, explicit `epsilon` |
| `haze` | `[beta,A]` | `x*exp(-beta*depth)+A*(1-exp(-beta*depth))`; `beta >= 0`, scalar `A` in [0,1], fixed nonnegative depth |
| `blur_noise` | `[sigma_blur,sigma_noise]` | Explicit `blur_then_noise` or `noise_then_blur` order |
| `blur_haze` | `[sigma_blur,beta,A]` | Explicit `blur_then_haze` or `haze_then_blur` order |

The configuration validates the blur domain/support. Tensor-level operator functions assume physical parameter-domain constraints are satisfied; they avoid data-dependent Python checks on parameters so that JVPs and higher derivatives remain usable. Invalid widths, depth or physical values should not be passed to these functions.

### Gaussian blur

The isotropic implementation uses separable horizontal/vertical convolution. The anisotropic implementation uses a full 2-D kernel. Both normalize their kernels and keep the support fixed at 33 × 33 across the full default width interval [0.5,3.5]. The radius of 16 pixels covers more than four times the maximum width. This is still a finite sampled approximation; checking sensitivity to larger support belongs in later derivative/geometry validation.

The covariance is `R(theta) diag(sigma_x²,sigma_y²) R(theta)^T`. Image coordinates use x rightward and y downward, so positive theta rotates toward the downward image axis. The chart has equivalences:

- `theta` and `theta + pi` describe the same kernel.
- Swapping widths and adding `pi/2` describes the same kernel.
- At equal widths, orientation is locally unidentifiable.

These facts matter later: a pullback metric can be semidefinite, rather than positive definite, at redundant parameterizations. Do not hide this using a numerical diagonal regularizer and then infer intrinsic curvature. Initially use fixed orientation for the two-width study; establish an identifiable chart before the full three-parameter analysis.

Exactly zero blur width is outside the differentiable sampled-Gaussian chart. Use the actual clean image for a clean reference. Selecting an identity anchor/chart for the eventual restoration flow remains a separate design decision.

### Crop boundaries

Reflection padding preserves output shape. The metric-study protocol removes its effect from the measured core: apply the operator to a source-image context crop, then call `core_crop`. For a single default blur, the 16-pixel halo covers the entire kernel radius. If later compositions include **multiple blur passes**, increase the halo to cover the cumulative support.

### Noise

Use the same `epsilon` when comparing parameter values and when differentiating/finite-differencing an operator. Resampling would mix operator sensitivity with random variation. `make_epsilon` uses a local CPU generator and does not reset global RNG state. Saved realizations are authoritative for exact replay across software-version changes. Task 3 must eventually average over multiple noise realizations.

### Haze

DIV2K supplies no depth in this project. The previews therefore show both unit depth and an explicitly synthetic horizontal depth ramp. Unit-depth haze is global attenuation toward airlight. The ramp demonstrates spatial variation and operator order; it is not true scene depth and cannot justify a claim about realistic haze geometry. Decide a depth-source/protocol before interpreting blur+haze results scientifically.

Airlight is achromatic and scalar here. Modeling RGB airlight would add parameters and can be introduced explicitly later. At `beta=0`, airlight is unidentifiable because the haze operator is identity.

### Color representation

Default blur and noise act in sRGB-coded values, matching a controlled digital image operator. This is a deliberate convention, not a claim that it is a physical optical model. `sampling.color_space: linear_rgb` applies the standard sRGB inverse transfer function before degrading and converts only the previews back to sRGB. Switching representations changes the ambient inner product and potentially the induced metric; use a new experiment directory and compare explicitly rather than pooling them.

## 8. Read the code in this order

1. `configs/default.yaml`: freeze the sampling and parameter units.
2. `geometry_lab/datasets.py`: understand the image IDs, crop coordinates, color conversion and halo.
3. `geometry_lab/prepare.py`: follow the dataset checks and output files.
4. `geometry_lab/degradations/common.py`: inspect how tensors/parameters are handled without detaching gradients.
5. `intensity.py` and `gaussian_noise.py`: simplest operators with known analytic derivatives.
6. `gaussian_blur.py` and `anisotropic_blur.py`: inspect kernel coordinates, normalization and fixed support.
7. `haze.py` and `composition.py`: inspect explicit auxiliary inputs and order.
8. `geometry_lab/preview.py` and `tests/test_task01.py`: see usage and correctness checks.

Minimal future-use example:

```python
import torch
from geometry_lab.datasets import ManifestDataset, core_crop
from geometry_lab.degradations import gaussian_blur

ds = ManifestDataset("outputs/task01/train_manifest.json")
item = ds[0]
x = item["image"]  # Includes physical context.
xi = torch.tensor([1.5], dtype=x.dtype, requires_grad=True)
y_context = gaussian_blur(x, xi, kernel_size=33)
y_core = core_crop(y_context, item["halo"], item["crop_size"])
# Later Task 2 differentiates the mapping xi -> y_core.
```

## 9. Completion criteria and next task

Task 0 is locally complete when your source-image counts/hashes are validated, both crop manifests are generated and the sampling protocol is fixed. Task 1 is locally complete when the progression grids look correct and the synthetic software checks pass in your environment.

Next is **Task 2: full Jacobian estimator and finite-difference validation**, including step-size sweeps, precision checks, support sensitivity and fixed stochastic realizations. The current gradient/JVP smoke checks establish graph connectivity and simple identities; they do not replace that study.

The later target mapping is `F_x(xi) = core_crop(D_xi(context_x))`. Its normalized pullback matrix is `J_x^T J_x / (3 * crop_size²)`, averaged over crops within each image and then over images. It depends on the chosen clean-image sampling distribution, color representation and ambient inner product. Tasks 0 and 1 make those choices explicit; they do not establish stability, curvature or restoration benefit.

## References for the implementation

- PyTorch convolution: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
- PyTorch padding: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
- PyTorch JVP: https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html
