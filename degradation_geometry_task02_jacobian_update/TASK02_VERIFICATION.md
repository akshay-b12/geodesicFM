# Verification of the Task 2 Jacobian update

Verified on 2026-10-01 in a hosted Linux CPU environment, using Python 3.12.14,
PyTorch 2.7.1+cpu, NumPy 2.3.5 and the existing Task 0/1 operators.

## Automated tests

**44 tests passed**: the original 23 Task 0/1 tests and 21 new Jacobian/runner tests.

The new tests cover:

- Parameter-first layout and a tiny full reverse-Jacobian reference.
- Known polynomial derivatives and second-order finite-difference convergence.
- Second-order forward/backward boundary stencils.
- Explicit unavailable stencils and unresolvable float32 perturbations.
- Correct treatment of float32 decimal domain endpoints.
- Near-zero derivative gates, nonfinite outputs and shape/input errors.
- Rejection of an isolated best finite-difference step.
- Independent reverse-mode projections, including detection of a wrong Jacobian.
- Higher-order graph preservation through polynomial and blur Jacobians.
- Closed-form/chain-rule references for every registered operator family.
- Zero orientation derivative at equal blur widths and the isotropic width identity.
- Negative controls that detach parameters or resample noise.
- Report generation, refusal to overwrite a run, all four stage integrations,
  unavailable physical halo handling and explicit image-index selection.

## Complete synthetic controls

Using the default eleven operator families, three patterns, two noise realizations
for stochastic families and ten epsilon values:

| Quantity | Result |
|---|---|
| Cases | 126 |
| Failed cases | 0 |
| Per-coordinate finite-difference rows | 2,550 |
| Overall status | PASS |

This run includes independent closed-form normalized Gaussian derivatives and
composition chain rules, in addition to full-image finite-difference sweeps and
reverse projections. It verifies the tested numerical implementations; it does
not establish geometry claims.

## Synthetic manifest integration

The existing demo pipeline was used with two training images and one validation
image, each with a 128 × 128 fixed core, and the default 33 × 33 kernel.

| Stage | Coverage | Result |
|---|---|---|
| Float64 pilot | 126 cases, 2,550 FD rows | PASS, zero failed cases |
| Float32 precision | 42 cases, 85 AD32-versus-AD64 coordinate comparisons | REVIEW_REQUIRED |
| Kernel support | One eligible core, seven blur-containing families, 63 coordinate comparisons of 33 versus 49 support | PASS |

In the float32 stage, all 85 AD32-versus-AD64 comparisons met the configured
tolerance. Three finite-difference cases failed to obtain two adjacent passing
steps for a narrow blur-width coordinate. The affected settings were:

- Fixed-orientation anisotropic blur at `(sigma_x,sigma_y)=(0.5,2.0)`.
- Fixed-orientation anisotropic blur at `(3.5,0.5)`.
- Three-parameter anisotropic blur at `(3.5,0.5,0)`.

The stage correctly preserves these review flags rather than weakening the
tolerances or treating them as float64 correctness failures. Diagnostic derivative
maps and sweep plots were visually inspected; plot label layout was adjusted.

## Practical scope

The actual DIV2K files on the user's computer were not available here. The user
must run the controls and DIV2K pilot in their configured environment. Windows,
CUDA, the default 256 × 256 DIV2K pilot and a full 900-image metric study were not
executed in this development environment. The fixture integration uses saved
manifests and the same numerical code path, but is not a natural-image result.

This is an incremental source update. It contains no replacement for the user's
default config, original operators, image manifests or previous experiment outputs.
