# Task 4 verification

## Package and compatibility

The update adds ten files: `configs/task04.yaml`; six files under `task04_analysis` (`__init__.py`, `__main__.py`, `io.py`, `numerics.py`, `plots.py`, `report.py`); `tests/test_task04.py`; `TASK04_README.md`; and this verification document.

No existing Task 0–3 files are replaced. A SHA256 inventory confirmed that every `.py` file under `geometry_lab` remained unchanged and that no new file appeared there. This preserves Task 3's code-hash scope and resume behavior. Task 4's imports and outputs are separate from Task 3's source tree. A subprocess check confirmed that importing the Task 4 CLI does not import PyTorch.

Verified on Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.10.8 on Linux CPU. PyTorch 2.7.1+cpu was used only to execute the earlier project's regression tests. Task 4 itself uses NumPy, PyYAML and Matplotlib already available in the Task 2/3 environment. Windows execution was not available here; the documented commands use the same portable Python module entry point.

## Tests

All **82 tests passed**, including the existing 62 tests and 20 new Task 4 tests. New coverage verifies:

- Exact rank-deficient matrices, signed derivative correlations and undefined zero-direction correlations.
- Zero matrices and repeated eigenvalues reporting a subspace rather than an arbitrary weak vector.
- Principal-angle/projector invariance to eigenvector signs and basis rotations.
- Orthogonal weak directions and explicit unavailability when compared subspace dimensions change.
- Preservation of roundoff-negative eigenvalues; rejection of substantial PSD, symmetry and finite-value failures.
- Raw per-image PSD violations cannot be hidden by a smaller characteristic scale.
- The difference between locally normalized rank and grid-wide-reference rank, including weak one-dimensional signals whose local condition remains one.
- Rank-threshold sensitivity and fixed coordinate scaling.
- Axial grid edges and an actual sampled reference point.
- Increased rank on averaging different image matrices; unavailable sample standard deviation with one image.
- Rejection of duplicate points, invalid scales, float32 inputs, mean mismatches, changed grids/image IDs and invalid source fingerprints.
- Read-only end-to-end analysis, separate split comparison, controlled refresh and refusal to write into the source tree.
- Missing aggregates rejected before output creation.
- Zero-metric plot paths and controls payload consistency.

## Integration and visual checks

The CLI analyzed the existing **synthetic pilot image run** over all default parameter grids: two synthetic training images, one synthetic validation image, and one crop each. It completed eight operator/split analyses covering intensity (9 points), additive noise (4), isotropic blur (9) and fixed-angle anisotropic blur (25). It also imported the Task 3 synthetic controls: **207 cases across 33 operator/pattern groups**, all with passing upstream analytic gates.

The pilot analysis produced JSON/CSV diagnostics, split comparisons, a readable report and **28 PNG figures plus 28 PDF figures**. Representative anisotropic identifiability maps and rank-sensitivity plots were visually inspected. Rank colors are discrete; undefined correlations are grey; nonpositive eigenvalues remain in the numeric outputs and are explicitly counted when omitted from logarithmic plots. An earlier integration run also analyzed all crops of the existing synthetic full image run.

These are synthetic execution and arithmetic checks. The user's DIV2K pilot matrices have not yet been supplied, so there is no new conclusion about their DIV2K metric, rank stability or geometry in this package. Pilot interpretation remains exploratory. Task 5 uncertainty/stability and Tasks 6–10 geometry verification remain separate.

## Reproduce

```powershell
python -m unittest discover -s tests -p test_task04.py -v
python -m task04_analysis --run outputs/task03/pilot --controls-run outputs/task03/controls --output outputs/task04/pilot
```

See `TASK04_README.md` for input paths, the upload checklist, analysis definitions and full CLI examples. The default output directory must be separate from the Task 3 run. Use a new directory for changed settings/inputs, or `--refresh` only for the same analysis identity.
