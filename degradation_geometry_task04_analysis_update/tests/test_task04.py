"""Independent mathematical controls, strict imports and read-only integration."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from task04_analysis.io import load_settings, load_run, load_matrices, fingerprint, atomic_json, load_controls, sha256
from task04_analysis.numerics import inspect_matrix, analyze_field, subspace_change, describe_images
from task04_analysis.__main__ import analyze

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = load_settings(ROOT/"configs/task04.yaml")


def fixture(root):
    root = Path(root)
    grid = {"points": [[0.], [.1]], "scales": [.1], "parameter_names": ["sigma_noise"],
            "axes": [[0., .1]], "bounds": [[0., .2]], "fixed_theta_rad": None}
    selections = {split: {"selected": [{"image": {"image_id": f"{split}/{i}"}, "records": [{"crop_index": 0}]}
                                     for i in range(2 if split == "train" else 1)]} for split in ("train", "valid")}
    plan = {"stage": "pilot", "grids": {"gaussian_noise": grid}, "selections": selections}
    atomic_json(root/"run.json", {"schema_version": 1, "fingerprint": fingerprint(plan), "plan": plan})
    for split, values in (("train", [1., 3.]), ("valid", [5.])):
        per_image = np.repeat(np.asarray(values)[:, None, None, None], 2, axis=1)
        destination = root/"analysis/gaussian_noise"/split
        destination.mkdir(parents=True)
        np.savez_compressed(destination/"metric.npz", points=np.asarray(grid["points"]), scales=np.asarray(grid["scales"]),
                            mean_metric=per_image.mean(0), per_image_metric=per_image,
                            image_ids=np.asarray([f"{split}/{i}" for i in range(len(values))]), crop_counts=np.ones(len(values), dtype=int))
    return root


class Task04MathTests(unittest.TestCase):
    def test_rank_and_signed_correlation(self):
        d = inspect_matrix(np.array([[1., -1.], [-1., 1.]]), SETTINGS)
        self.assertEqual(d["rank"], 1)
        self.assertAlmostEqual(d["correlation"][0][1], -1)
        self.assertIsNone(d["condition_number"])
        self.assertAlmostEqual(d["participation_ratio"], 1)

    def test_zero_metric_has_no_arbitrary_weak_vector(self):
        d = inspect_matrix(np.zeros((2, 2)), SETTINGS)
        self.assertEqual(d["rank"], 0)
        self.assertEqual(d["weak_subspace_dimension"], 2)
        self.assertIsNone(d["unique_weak_vector"])
        self.assertIsNone(d["correlation"][0][0])
        self.assertIsNone(d["participation_ratio"])

    def test_repeated_eigenvalue_reports_subspace(self):
        d = inspect_matrix(np.diag([1., 1., 5.]), SETTINGS)
        self.assertEqual(d["weak_subspace_dimension"], 2)
        self.assertIsNone(d["unique_weak_vector"])

    def test_sign_and_basis_invariant_subspace_comparison(self):
        a = inspect_matrix(np.diag([1., 1., 5.]), SETTINGS)
        b = copy.deepcopy(a)
        basis = np.asarray(b["weak_basis_columns"])
        b["weak_basis_columns"] = (basis@np.array([[0., -1.], [1., 0.]])).tolist()
        change = subspace_change(a, b)
        self.assertAlmostEqual(change["normalized_projector_distance"], 0)
        self.assertAlmostEqual(change["max_principal_angle_degrees"], 0)

    def test_orthogonal_weak_directions_and_dimension_change(self):
        a = inspect_matrix(np.diag([.01, 1.]), SETTINGS)
        b = inspect_matrix(np.diag([1., .01]), SETTINGS)
        d = subspace_change(a, b)
        self.assertAlmostEqual(d["max_principal_angle_degrees"], 90)
        self.assertAlmostEqual(d["normalized_projector_distance"], 1)
        self.assertFalse(subspace_change(a, inspect_matrix(np.eye(2), SETTINGS))["comparable_weak_spaces"])

    def test_roundoff_negative_retained_and_invalid_psd_rejected(self):
        g = np.diag([1., -1e-15])
        before = g.copy()
        d = inspect_matrix(g, SETTINGS)
        np.testing.assert_array_equal(g, before)
        self.assertLess(d["eigenvalues"][0], 0)
        self.assertEqual(d["negative_roundoff_eigenvalues"], 1)
        for matrix in (np.diag([1., -.01]), np.array([[1., .2], [.1, 1.]]), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                inspect_matrix(matrix, SETTINGS)

    def test_local_and_global_rank_differ_for_weak_one_dimensional_metric(self):
        settings = {**SETTINGS, "rank_atol": 0.}
        field = analyze_field([[.5], [2.]], np.array([[[1.]], [[1e-10]]]), [1.], settings)
        self.assertEqual([d["scaled"]["rank"] for d in field["diagnostics"]], [1, 1])
        self.assertEqual([d["scaled"]["global_reference_rank"] for d in field["diagnostics"]], [1, 0])
        self.assertEqual(field["diagnostics"][1]["scaled"]["condition_number"], 1)

    def test_threshold_sweep_and_scaling(self):
        g = np.diag([1., 1e-8])
        field = analyze_field([[1., 0.]], g[None], [1., .1], SETTINGS)
        np.testing.assert_allclose(field["diagnostics"][0]["scaled"]["eigenvalues"], [1e-10, 1.])
        ranks = [d["ranks"][0] for d in field["rank_sweep"]]
        self.assertEqual(field["diagnostics"][0]["scaled"]["rank"], 1)
        field = analyze_field([[1., 0.]], g[None], [1., 1.], SETTINGS)
        ranks = [d["ranks"][0] for d in field["rank_sweep"]]
        self.assertIn(1, ranks)
        self.assertIn(2, ranks)

    def test_axial_edges_and_reference_are_actual_sampled_points(self):
        points = [[0., 0.], [0., 2.], [1., 0.], [1., 2.]]
        field = analyze_field(points, np.repeat(np.eye(2)[None], 4, axis=0), [1., 2.], SETTINGS)
        self.assertEqual(len(field["edges"]), 4)
        self.assertTrue(all(e["scaled_step"] == 1 for e in field["edges"]))
        self.assertTrue(all(e["relative_change"] == 0 for e in field["edges"]))
        self.assertIn(field["reference_point"], points)

    def test_image_rank_can_increase_on_averaging(self):
        values = np.array([np.diag([1., 0.]), np.diag([0., 1.])])[:, None]
        field = analyze_field([[1., 1.]], values.mean(0), [1., 1.], SETTINGS)
        images = describe_images(values, field, [1., 1.], SETTINGS)
        self.assertEqual(field["diagnostics"][0]["scaled"]["rank"], 2)
        self.assertEqual(images[0]["scaled_image_ranks"], [1, 1])
        self.assertEqual(images[0]["scaled_image_full_rank_fraction"], 0)

    def test_single_image_std_is_unavailable(self):
        g = np.array([[[1.]]])
        f = analyze_field([[1.]], g, [1.], SETTINGS)
        self.assertIsNone(describe_images(g[None], f, [1.], SETTINGS)[0]["image_std_raw_entries"])

    def test_image_raw_psd_violation_not_hidden_by_coordinate_scaling(self):
        values = np.array([[[[-1e-12]]], [[[3e-12]]]])
        field = analyze_field([[0.]], values.mean(0), [.1], SETTINGS)
        with self.assertRaises(ValueError):
            describe_images(values, field, [.1], SETTINGS)

    def test_duplicate_points_and_invalid_scales_rejected(self):
        for points, scales in (([[1.], [1.]], [1.]), ([[1.], [2.]], [0.])):
            with self.assertRaises(ValueError):
                analyze_field(points, np.ones((2, 1, 1)), scales, SETTINGS)


class Task04IOTests(unittest.TestCase):
    def test_numpy_only_import(self):
        result = subprocess.run([sys.executable, "-c", "import task04_analysis.__main__,sys; assert 'torch' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_aggregate_validation_and_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = fixture(Path(temporary)/"source")
            root, run = load_run(source)
            item = load_matrices(root, run, "gaussian_noise", "train")
            self.assertEqual(float(item["arrays"]["mean_metric"][0, 0, 0]), 2.)
            changed = copy.deepcopy(run)
            changed["plan"]["stage"] = "full"
            atomic_json(source/"run.json", changed)
            with self.assertRaises(ValueError):
                load_run(source)

    def test_reject_float32_wrong_mean_points_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = fixture(Path(temporary)/"source")
            root, run = load_run(source)
            path = source/"analysis/gaussian_noise/train/metric.npz"
            with np.load(path) as data:
                original = {key: data[key] for key in data.files}
            for key, wrong in (("mean_metric", original["mean_metric"].astype(np.float32)),
                               ("mean_metric", original["mean_metric"]+1.),
                               ("points", np.array([[0.], [.2]])),
                               ("image_ids", np.array(["train/0", "train/other"]))):
                arrays = {**original, key: wrong}
                np.savez_compressed(path, **arrays)
                with self.assertRaises(ValueError):
                    load_matrices(root, run, "gaussian_noise", "train")

    def test_read_only_end_to_end_refresh_and_split_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = fixture(Path(temporary)/"source")
            original = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
            out = Path(temporary)/"analysis"
            settings = {**SETTINGS, "plots": False}
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(source, out, settings)
                analyze(source, out, settings, refresh=True)
            self.assertAlmostEqual(result["split_comparisons"]["gaussian_noise"]["max_relative_difference"], .6)
            self.assertTrue((out/"REPORT.md").is_file())
            self.assertEqual(original, {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()})
            with self.assertRaises(ValueError):
                analyze(source, source/"analysis/task04", settings)
            with self.assertRaises(ValueError):
                analyze(source, out, {**settings, "rank_atol": 1e-9}, refresh=True)

    def test_missing_aggregate_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = fixture(Path(temporary)/"source")
            (source/"analysis/gaussian_noise/train/metric.npz").unlink()
            output = Path(temporary)/"out"
            with self.assertRaises(ValueError):
                analyze(source, output, {**SETTINGS, "plots": False})
            self.assertFalse(output.exists())

    def test_zero_and_repeated_eigenvalue_plot_paths(self):
        from task04_analysis.plots import plot_field
        with tempfile.TemporaryDirectory() as temporary:
            points = [[.5, .5], [.5, 1.], [1., .5], [1., 1.]]
            grid = {"points": points, "scales": [1., 1.], "parameter_names": ["x", "y"], "axes": [[.5, 1.], [.5, 1.]]}
            field = analyze_field(points, np.zeros((4, 2, 2)), [1., 1.], SETTINGS)
            plot_field(temporary, grid, field, "zero_control", "train")
            self.assertTrue((Path(temporary)/"identifiability_grid.png").is_file())

    def test_controls_input_consistency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {"stage": "controls"}
            atomic_json(root/"run.json", {"schema_version": 1, "fingerprint": fingerprint(plan), "plan": plan})
            row = {"operator": "gaussian_noise", "pattern": "constant", "point": [0.], "metric": [[1.]],
                   "diagnostics": {"scales": [.1]}, "passed": True, "checks": {"analytic_gram": True}, "max_abs_gram_error": 0.}
            atomic_json(root/"controls.json", {"stage": "controls", "cases": 1, "failed": 0, "rows": [row]})
            self.assertEqual(len(load_controls(root)[2]), 1)
            row["checks"]["analytic_gram"] = False
            atomic_json(root/"controls.json", {"stage": "controls", "cases": 1, "failed": 0, "rows": [row]})
            with self.assertRaises(ValueError):
                load_controls(root)


if __name__ == "__main__":
    unittest.main()
