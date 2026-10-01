"""Scientific implementation checks using small synthetic images, not DIV2K."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from geometry_lab.config import load_config
from geometry_lab.datasets import (ManifestDataset, build_split_manifest, core_crop,
                                   rgb_tensor, stable_seed)
from geometry_lab.degradations import (affine_intensity, gaussian_blur, gaussian_kernel1d,
    anisotropic_blur, anisotropic_kernel, gaussian_noise, make_epsilon, haze,
    blur_noise, blur_haze)
from geometry_lab.prepare import prepare
from geometry_lab.preview import preview


torch.set_num_threads(2)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.x = torch.rand((3, 17, 19), generator=generator, dtype=torch.float64)
        self.epsilon = make_epsilon(self.x, 1234)
        self.depth = torch.linspace(0.2, 1.8, 19, dtype=self.x.dtype)[None, None, :].expand(1, 17, 19)

    def test_blur_kernel_normalization_and_symmetry(self):
        kernel = gaussian_kernel1d(self.x, [1.0], 7)
        torch.testing.assert_close(kernel.sum(), torch.tensor(1.0, dtype=self.x.dtype))
        torch.testing.assert_close(kernel, kernel.flip(0))
        aniso = anisotropic_kernel(self.x, [1.2, 0.7, 0.8], 7)
        torch.testing.assert_close(aniso.sum(), torch.tensor(1.0, dtype=self.x.dtype))
        torch.testing.assert_close(aniso, aniso.flip((0, 1)))

    def test_blur_preserves_constant_images(self):
        x = torch.ones_like(self.x) * 0.37
        torch.testing.assert_close(gaussian_blur(x, [1.0], 7), x)
        torch.testing.assert_close(anisotropic_blur(x, [1.3, 0.8, 0.5], 7), x)

    def test_equal_axes_match_isotropic(self):
        for theta in (0.0, 0.7, 1.5):
            torch.testing.assert_close(anisotropic_blur(self.x, [1.0,1.0,theta], 7),
                                       gaussian_blur(self.x, [1.0], 7))

    def test_orientation_equivalences(self):
        p = [1.4, 0.7, 0.4]
        kernel = anisotropic_kernel(self.x, p, 7)
        torch.testing.assert_close(kernel, anisotropic_kernel(self.x, [p[0],p[1],p[2]+math.pi], 7))
        torch.testing.assert_close(kernel, anisotropic_kernel(self.x, [p[1],p[0],p[2]+math.pi/2], 7))

    def test_rotation_changes_anisotropic_kernel(self):
        a = anisotropic_kernel(self.x, [1.4,0.7,0.0], 7)
        b = anisotropic_kernel(self.x, [1.4,0.7,math.pi/2], 7)
        torch.testing.assert_close(a.T, b)
        self.assertGreater(float((a-b).abs().max()), 0.01)

    def test_affine_identity_and_no_clipping(self):
        torch.testing.assert_close(affine_intensity(self.x, [1.0,0.0]), self.x)
        y = affine_intensity(self.x, [3.0,-0.4])
        self.assertLess(float(y.min()), 0)
        self.assertGreater(float(y.max()), 1)

    def test_affine_analytic_jacobian(self):
        p = torch.tensor([1.2,0.1], dtype=self.x.dtype, requires_grad=True)
        for tangent, expected in (([1.0,0.0], self.x), ([0.0,1.0], torch.ones_like(self.x))):
            _, j = torch.func.jvp(lambda xi: affine_intensity(self.x, xi), (p,),
                                  (torch.tensor(tangent, dtype=self.x.dtype),))
            torch.testing.assert_close(j, expected)

    def test_noise_explicit_reproducibility_and_identity(self):
        torch.testing.assert_close(make_epsilon(self.x, 1234), self.epsilon, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.epsilon, make_epsilon(self.x, 1235)))
        torch.testing.assert_close(gaussian_noise(self.x, [0.0], epsilon=self.epsilon), self.x)
        torch.testing.assert_close(gaussian_noise(self.x, [0.1], epsilon=self.epsilon)-self.x, 0.1*self.epsilon)

    def test_noise_analytic_jacobian(self):
        p = torch.tensor([0.1], dtype=self.x.dtype)
        _, j = torch.func.jvp(lambda xi: gaussian_noise(self.x, xi, epsilon=self.epsilon), (p,), (torch.ones_like(p),))
        torch.testing.assert_close(j, self.epsilon)

    def test_haze_identity_and_formula(self):
        torch.testing.assert_close(haze(self.x, [0,0.9], depth=self.depth), self.x)
        t = torch.exp(-0.5*self.depth)
        torch.testing.assert_close(haze(self.x, [0.5,0.9], depth=self.depth), self.x*t+0.9*(1-t))

    def test_composition_orders_are_explicit(self):
        a = blur_noise(self.x, [1.0,0.1], epsilon=self.epsilon, kernel_size=7)
        torch.testing.assert_close(a, gaussian_blur(self.x,[1.0],7)+0.1*self.epsilon)
        b = blur_noise(self.x, [1.0,0.1], epsilon=self.epsilon, kernel_size=7, order="noise_then_blur")
        self.assertGreater(float((a-b).square().mean()), 0.001)
        h1 = blur_haze(self.x, [1,0.7,0.9], depth=self.depth, kernel_size=7)
        h2 = blur_haze(self.x, [1,0.7,0.9], depth=self.depth, kernel_size=7, order="haze_then_blur")
        self.assertGreater(float((h1-h2).abs().max()), 1e-5)

    def test_batched_and_unbatched_agree(self):
        batch = torch.stack((self.x, self.x*0.8))
        for fn in (lambda x: gaussian_blur(x,[1],7), lambda x: anisotropic_blur(x,[1,0.7,0.4],7)):
            y = fn(batch)
            self.assertEqual(y.shape, batch.shape)
            torch.testing.assert_close(y[0], fn(batch[0]))
            torch.testing.assert_close(y[1], fn(batch[1]))

    def test_halo_core_matches_full_image_blur(self):
        # Avoid artificial reflection boundaries using original-image context.
        x = torch.rand((3,40,40), generator=torch.Generator().manual_seed(5), dtype=self.x.dtype)
        top,left,size,h = 9,8,12,3
        patch = x[:,top-h:top+size+h,left-h:left+size+h]
        for fn in (lambda z: gaussian_blur(z,[1],7), lambda z: anisotropic_blur(z,[1,0.7,0.4],7)):
            torch.testing.assert_close(core_crop(fn(patch),h,size), fn(x)[:,top:top+size,left:left+size])

    def test_all_operators_preserve_parameter_gradients_and_jvp(self):
        cases = [
            (lambda p: gaussian_blur(self.x,p,7), [1.0]),
            (lambda p: anisotropic_blur(self.x,p,7), [1.2,0.7,0.4]),
            (lambda p: affine_intensity(self.x,p), [1.0,0.1]),
            (lambda p: gaussian_noise(self.x,p,epsilon=self.epsilon), [0.05]),
            (lambda p: haze(self.x,p,depth=self.depth), [0.5,0.9]),
            (lambda p: blur_noise(self.x,p,epsilon=self.epsilon,kernel_size=7), [1,0.05]),
            (lambda p: blur_haze(self.x,p,depth=self.depth,kernel_size=7), [1,0.5,0.9]),
        ]
        for fn, values in cases:
            with self.subTest(values=values):
                p = torch.tensor(values,dtype=self.x.dtype,requires_grad=True)
                y = fn(p)
                g, = torch.autograd.grad(y.square().mean(),p)
                self.assertTrue(torch.isfinite(g).all())
                self.assertTrue((g.abs() > 1e-10).all())
                _, j = torch.func.jvp(fn,(p,),(torch.ones_like(p),))
                self.assertTrue(torch.isfinite(j).all())
                self.assertEqual(j.shape,self.x.shape)

    def test_reject_invalid_shapes_and_support(self):
        with self.assertRaises(ValueError):
            gaussian_blur(self.x,[1],8)
        with self.assertRaises(ValueError):
            gaussian_blur(self.x,[1,2],7)
        with self.assertRaises(ValueError):
            gaussian_noise(self.x,[0.1],epsilon=self.epsilon[:,:-1])
        with self.assertRaises(ValueError):
            haze(self.x,[0.5,0.9],depth=torch.ones(2,17,19,dtype=self.x.dtype))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.valid = self.root / "valid"
        self.train.mkdir()
        self.valid.mkdir()
        for folder, name, seed in ((self.train,"0001.png",1),(self.train,"0002.png",2),(self.valid,"0801.png",3)):
            array = np.random.default_rng(seed).integers(0,256,(40,45,3),dtype=np.uint8)
            Image.fromarray(array).save(folder/name)
        self.sampling = {"seed":123, "crop_size":16, "crops_per_image":3, "halo":3, "color_space":"srgb"}

    def tearDown(self):
        self.temp.cleanup()

    def test_repeatable_crops_and_image_ids(self):
        a,_ = build_split_manifest(self.train,"train",self.sampling,2)
        b,_ = build_split_manifest(self.train,"train",self.sampling,2)
        self.assertEqual(a,b)
        self.assertEqual(len(a["records"]),6)
        self.assertEqual(len({r["image_id"] for r in a["records"]}),2)
        self.assertNotEqual(stable_seed(123,"train/0001",0),stable_seed(123,"train/0001",1))

    def test_manifest_dataset_and_image_fingerprint(self):
        m,_ = build_split_manifest(self.train,"train",self.sampling,2)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(m),encoding="utf-8")
        ds = ManifestDataset(path)
        item = ds[0]
        self.assertEqual(item["image"].shape,(3,22,22))
        self.assertEqual(core_crop(item["image"],3,16).shape,(3,16,16))
        self.assertEqual(item["image"].dtype,torch.float64)
        Image.fromarray(np.zeros((40,45,3),dtype=np.uint8)).save(self.train/"0001.png")
        with self.assertRaises(ValueError):
            ManifestDataset(path)

    def test_reject_bad_counts_small_images_and_non_rgb(self):
        with self.assertRaises(ValueError):
            build_split_manifest(self.train,"train",self.sampling,800)
        Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(self.train/"0003.png")
        with self.assertRaises(ValueError):
            build_split_manifest(self.train,"train",self.sampling,3)
        (self.train/"0003.png").unlink()
        Image.fromarray(np.zeros((40,45),dtype=np.uint8)).save(self.train/"0003.png")
        with self.assertRaises(ValueError):
            build_split_manifest(self.train,"train",self.sampling,3)

    def test_color_space_is_explicit(self):
        array = np.full((4,4,3),128,dtype=np.uint8)
        srgb = rgb_tensor(array,"srgb")
        linear = rgb_tensor(array,"linear_rgb")
        self.assertGreater(float(srgb.mean()),float(linear.mean()))
        self.assertAlmostEqual(float(linear.mean()),0.21586050011389926,places=12)

    def test_adding_image_does_not_change_existing_coordinates(self):
        a,_ = build_split_manifest(self.train,"train",self.sampling,2)
        array = np.random.default_rng(10).integers(0,256,(40,45,3),dtype=np.uint8)
        Image.fromarray(array).save(self.train/"0000.png")
        b,_ = build_split_manifest(self.train,"train",self.sampling,3)
        for old,new in zip(a["records"],b["records"][3:]):
            self.assertEqual((old["top"],old["left"]),(new["top"],new["left"]))

    def small_config(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/"configs"/"default.yaml")
        cfg["sampling"] = self.sampling
        cfg["dataset"].update(train_hr=str(self.train),valid_hr=str(self.valid),expected_train_images=2,expected_valid_images=1)
        cfg["operators"].update(kernel_size=7,sigma_domain=[0.5,0.7])
        cfg["experiment"]["output_dir"] = str(self.root/"artifacts")
        return cfg

    def test_prepare_and_preview_end_to_end(self):
        cfg = self.small_config()
        output = prepare(cfg)
        dest = preview(cfg)
        self.assertTrue((output/"crop_coordinates.json").exists())
        self.assertEqual(len(list(dest.glob("*.png"))),11)
        raw = torch.load(dest/"raw_tensors.pt",weights_only=True)
        self.assertEqual(raw["isotropic_blur"].shape,(5,3,16,16))
        metadata = json.loads((dest/"preview_metadata.json").read_text())
        self.assertEqual(metadata["source_image_id"],"train/0001")
        with self.assertRaises(FileExistsError):
            prepare(cfg)
        with self.assertRaises(FileExistsError):
            preview(cfg)

    def test_split_leakage_is_rejected(self):
        (self.valid/"0801.png").write_bytes((self.train/"0001.png").read_bytes())
        with self.assertRaises(ValueError):
            prepare(self.small_config())

    def test_sampling_config_mismatch_is_rejected(self):
        cfg = self.small_config()
        prepare(cfg)
        cfg["sampling"] = copy.deepcopy(cfg["sampling"])
        cfg["sampling"]["color_space"] = "linear_rgb"
        with self.assertRaises(ValueError):
            preview(cfg)


if __name__ == "__main__":
    unittest.main()
