# Verification of the delivered Task 0 + Task 1 project

Verified on 2026-09-30 in a hosted Linux CPU environment:

| Component | Version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.3.5 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |

`python -m unittest discover -s tests -v`: **23 tests passed**.

The tests cover deterministic crops, unchanged coordinates when unrelated images
are added, image fingerprints, split leakage, dtype/color conversion, image-size
and format rejection, manifest/config consistency, operator formulas/identities,
anisotropic orientation equivalences, explicit composition order, batched inputs,
halo-core agreement with full-image blur, and parameter-gradient/JVP connectivity.
Analytic JVPs are checked for the affine and additive-noise controls.

The separate CLI demo successfully generated three synthetic source images,
prepared both split manifests, and saved eleven operator progression grids plus
raw tensors and metadata. A preview layout was visually inspected and caption
wrapping was corrected before packaging.

DIV2K was not supplied to this environment. Therefore no DIV2K experiment,
operator-induced metric, curvature result, or scientific go/no-go decision is
claimed. Windows execution and CUDA execution have not been tested here.

The full Jacobian finite-difference sweeps remain Task 2. The current smoke checks
are necessary implementation checks, not substitutes for that validation.
