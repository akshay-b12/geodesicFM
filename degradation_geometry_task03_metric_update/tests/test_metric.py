"""Mathematical controls and failure/recovery coverage for Task 3."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from geometry_lab.datasets import build_split_manifest, stable_seed
from geometry_lab.jacobian import jacobian_jvp
from geometry_lab.jacobian_cases import operator_specs, auxiliary_fields, bind_operator, synthetic_context, analytic_jacobian
from geometry_lab.metric import gram_metric, image_average, metric_diagnostics, neighbor_variation
from geometry_lab.metric_grids import load_metric_config
from geometry_lab.estimate_metric import (initialize_run, atomic_json, read_shard,
                                          write_shard, estimate_image, main)
from geometry_lab.metric_analysis import analyze_run

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class MetricTests(unittest.TestCase):
    def test_scalar_normalization_and_off_diagonal(self):
        j = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2)
        g = gram_metric(j).numpy()
        reference = np.array([[float((j[i]*j[k]).sum())/12 for k in range(2)] for i in range(2)])
        np.testing.assert_allclose(g, reference, rtol=0, atol=0)

    def test_float32_and_nonfinite_rejected(self):
        for j in (torch.ones(1, 2), torch.tensor([[float("nan")]], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                gram_metric(j)

    def test_realization_grams_must_precede_averaging(self):
        j = torch.ones((1, 3, 2, 2), dtype=torch.float64)
        average_grams = (gram_metric(j)+gram_metric(-j))/2
        self.assertEqual(float(average_grams[0, 0]), 1)
        self.assertEqual(float(gram_metric((j-j)/2)[0, 0]), 0)

    def test_equal_image_weights_with_unequal_crops(self):
        ids, images, mean = image_average(np.array([1., 3., 10.])[:, None, None], ["a", "a", "b"])
        self.assertEqual(ids, ["a", "b"])
        np.testing.assert_array_equal(images[:, 0, 0], [2, 10])
        self.assertEqual(float(mean[0, 0]), 6)

    def test_averaged_rank_can_exceed_individual_image_rank(self):
        a, b = np.diag([1., 0.]), np.diag([0., 1.])
        self.assertEqual(metric_diagnostics(a, [1, 1])["scaled"]["rank"], 1)
        self.assertEqual(metric_diagnostics(b, [1, 1])["scaled"]["rank"], 1)
        self.assertEqual(metric_diagnostics((a+b)/2, [1, 1])["scaled"]["rank"], 2)

    def test_scaled_coordinate_transform(self):
        g = np.array([[2., .3], [.3, 4.]])
        s = np.array([3., .1])
        d = metric_diagnostics(g, s)
        np.testing.assert_allclose(d["scaled"]["eigenvalues"], np.linalg.eigvalsh(s[:, None]*g*s[None, :]))
        self.assertEqual(d["scaled"]["rank"], 2)

    def test_singular_metric_preserved(self):
        g = np.diag([1., 0.])
        snapshot = g.copy()
        d = metric_diagnostics(g, [1, 1])
        np.testing.assert_array_equal(g, snapshot)
        self.assertEqual(d["raw"]["eigenvalues"], [0, 1])
        self.assertEqual(d["raw"]["rank"], 1)
        self.assertIsNone(d["raw"]["condition_number"])
        self.assertIsNone(d["raw"]["correlation"][1][1])
        self.assertEqual(metric_diagnostics(np.zeros((2, 2)), [1, 1])["scaled"]["rank"], 0)

    def test_rank_convention_is_explicit(self):
        g = np.diag([1., 1e-10])
        self.assertEqual(metric_diagnostics(g, [1, 1])["raw"]["rank"], 1)
        self.assertEqual(metric_diagnostics(g, [1, 1], rank_atol=0, rank_rtol=1e-12)["raw"]["rank"], 2)

    def test_roundoff_negative_is_reported_without_clamping(self):
        g = np.diag([1., -1e-15])
        d = metric_diagnostics(g, [1, 1])
        self.assertLess(d["raw"]["eigenvalues"][0], 0)
        self.assertEqual(d["raw"]["roundoff_negative_eigenvalues"], 1)

    def test_invalid_psd_symmetry_scales_rejected(self):
        for g, scales in ((np.diag([1., -.01]), [1, 1]),
                          (np.array([[1., .2], [.1, 1.]]), [1, 1]),
                          (np.eye(2), [1, 0])):
            with self.assertRaises(ValueError):
                metric_diagnostics(g, scales)

    def test_affine_closed_form_and_constant_rank(self):
        specs = operator_specs([.5, 3.5])
        for pattern in ("constant", "textured"):
            x = synthetic_context(pattern, 8, 16)
            fn = bind_operator(specs["affine_intensity"], x, halo=16, core_size=8, kernel_size=33,
                               fields=auxiliary_fields(x, 2))
            g = gram_metric(jacobian_jvp(fn, torch.tensor([1., 0.], dtype=torch.float64))).numpy()
            core = x[:, 16:24, 16:24]
            np.testing.assert_allclose(g, [[float(core.square().mean()), float(core.mean())], [float(core.mean()), 1.]])
            self.assertEqual(metric_diagnostics(g, [1, .1])["scaled"]["rank"], 1 if pattern == "constant" else 2)

    def test_exact_nulls_in_full_orientation_chart(self):
        x = synthetic_context("textured", 8, 16)
        spec = operator_specs([.5, 3.5])["anisotropic_blur"]
        fn = bind_operator(spec, x, halo=16, core_size=8, kernel_size=33, fields=auxiliary_fields(x, 3))
        g = gram_metric(jacobian_jvp(fn, torch.tensor([2., 2., .47], dtype=torch.float64))).numpy()
        np.testing.assert_array_equal(g[2], np.zeros(3))
        self.assertEqual(metric_diagnostics(g, spec.scales)["raw"]["rank"], 2)

    def test_noise_estimator_constant_across_parameter_grid_and_reproducible(self):
        array = np.zeros((40, 40, 3), dtype=np.uint8)
        rec = {"top": 16, "left": 16, "halo": 16, "crop_size": 8, "image_id": "train/x", "crop_index": 0}
        settings = {"noise_seed": 2, "noise_realizations": 3}
        spec = operator_specs([.5, 3.5])["gaussian_noise"]
        a = estimate_image(spec, [[0.], [.1], [.2]], array, [rec], {"color_space": "srgb"}, settings, 33, "cpu")
        b = estimate_image(spec, [[0.], [.1], [.2]], array, [rec], {"color_space": "srgb"}, settings, 33, "cpu")
        np.testing.assert_array_equal(a["image_metric"], b["image_metric"])
        np.testing.assert_array_equal(a["image_metric"], np.repeat(a["image_metric"][:1], 3, axis=0))
        self.assertGreater(float(a["image_metric"][0, 0, 0]), 0)

    def test_all_registered_estimators_match_averaged_analytic_grams(self):
        array = np.random.default_rng(10).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        rec = {"top": 16, "left": 16, "halo": 16, "crop_size": 8, "image_id": "train/x", "crop_index": 0}
        settings = {"noise_seed": 2, "noise_realizations": 2}
        x = torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float64)/255
        for name, spec in operator_specs([.5, 3.5]).items():
            with self.subTest(operator=name):
                actual = estimate_image(spec, spec.points, array, [rec], {"color_space": "srgb"}, settings, 33, "cpu")["image_metric"]
                references = []
                for r in range(2 if spec.has_noise else 1):
                    fields = auxiliary_fields(x, stable_seed("task03_noise", 2, "train/x", 0, r))
                    references.append(np.stack([gram_metric(analytic_jacobian(spec, x, torch.tensor(p, dtype=torch.float64),
                        halo=16, core_size=8, fields=fields, kernel_size=33)).numpy() for p in spec.points]))
                np.testing.assert_allclose(actual, np.stack(references).mean(0), rtol=1e-10, atol=1e-12)

    def test_neighbor_edges_and_scaled_steps(self):
        p = [[0., 0.], [0., 2.], [1., 0.], [1., 2.]]
        g = np.stack([np.eye(2)*a for a in (1., 1., 2., 2.)])
        edges = neighbor_variation(p, g, [1., 2.])
        self.assertEqual(len(edges), 4)
        self.assertTrue(all(e["scaled_step"] == 1 for e in edges))
        self.assertEqual(sum(e["relative_change"] > 0 for e in edges), 2)
        self.assertEqual(neighbor_variation([[0.], [1.]], np.zeros((2, 1, 1)), [1.])[0]["relative_change"], 0)

    def test_grid_bounds_duplicates_axis_order(self):
        original = yaml.safe_load((ROOT/"configs/metric.yaml").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"metric.yaml"
            path.write_text(yaml.safe_dump(original))
            _, grids, _ = load_metric_config(path, [.5, 3.5])
            self.assertEqual(grids["affine_intensity"]["points"][1], [.5, 0.])
            for bad in ([[.4]], [[1.], [1.]], [[float("nan")]]):
                cfg = copy.deepcopy(original)
                cfg["grids"] = {"isotropic_blur": {"points": bad}}
                path.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    load_metric_config(path, [.5, 3.5])

    def test_resume_identity_and_checkpoint_damage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"run"
            identity = initialize_run(directory, {"data": 1})
            self.assertEqual(initialize_run(directory, {"data": 1}, True), identity)
            for plan, resume in (({"data": 1}, False), ({"data": 2}, True)):
                with self.assertRaises(ValueError):
                    initialize_run(directory, plan, resume)
            path = directory/"shard.npz"
            arrays = {"image_metric": np.ones((2, 1, 1)), "crop_metrics": np.ones((1, 2, 1, 1)),
                      "realization_trace_std": np.zeros((1, 2)), "crop_indices": np.array([0])}
            write_shard(path, identity, "a", arrays)
            self.assertIsNotNone(read_shard(path, identity, "a", (2, 1, 1)))
            with self.assertRaises(ValueError):
                read_shard(path, identity, "b", (2, 1, 1))
            path.write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                read_shard(path, identity, "a", (2, 1, 1))
            path.with_suffix(".json").unlink()
            self.assertIsNone(read_shard(path, identity, "a", (2, 1, 1)))

    def test_end_to_end_split_means_resume_missing_shard_and_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"configs").mkdir()
            cfg = yaml.safe_load((ROOT/"configs/default.yaml").read_text())
            cfg["sampling"].update(crop_size=8, crops_per_image=2)
            cfg["experiment"]["output_dir"] = "prepared"
            for split, colors in (("train", (51, 102)), ("valid", (204,))):
                folder = root/split
                folder.mkdir()
                for i, color in enumerate(colors):
                    Image.fromarray(np.full((48, 48, 3), color, dtype=np.uint8)).save(folder/f"{i}.png")
                cfg["dataset"][f"{split}_hr"] = str(folder)
                cfg["dataset"][f"expected_{split}_images"] = len(colors)
                manifest, _ = build_split_manifest(folder, split, cfg["sampling"], len(colors))
                atomic_json(root/"prepared"/f"{split}_manifest.json", manifest)
            config = root/"configs/default.yaml"
            config.write_text(yaml.safe_dump(cfg))
            settings = yaml.safe_load((ROOT/"configs/metric.yaml").read_text())
            settings.update(output_dir="metrics", plots=False)
            settings["grids"] = {"affine_intensity": {"points": [[1., 0.], [1.5, .1]]}}
            metric_config = root/"configs/metric.yaml"
            metric_config.write_text(yaml.safe_dump(settings))
            args = ["--config", str(config), "--metric-config", str(metric_config), "--stage", "full"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
                main(args+["--resume"])
            run = root/"metrics/full"
            for split, expected in (("train", .3), ("valid", .8)):
                with np.load(run/f"analysis/affine_intensity/{split}/metric.npz") as data:
                    self.assertAlmostEqual(float(data["mean_metric"][0, 0, 1]), expected)
                    self.assertEqual(len(data["image_ids"]), 2 if split == "train" else 1)
            markers = sorted((run/"checkpoints").rglob("*.json"))
            markers[0].unlink()
            with self.assertRaises(ValueError):
                analyze_run(run, plots=False)
            with contextlib.redirect_stdout(io.StringIO()):
                main(args+["--resume"])
            Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8)).save(root/"train/0.png")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                main(args+["--resume"])


if __name__ == "__main__":
    unittest.main()
