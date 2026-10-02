"""Numerical mathematics and negative controls for the Jacobian foundation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from geometry_lab.config import load_config
from geometry_lab.datasets import build_split_manifest
from geometry_lab.jacobian import (jacobian_jvp,finite_difference_jacobian,
    derivative_error,adjacent_pass_window,reverse_projection_check)
from geometry_lab.jacobian_cases import (operator_specs,auxiliary_fields,
    bind_operator,analytic_jacobian,synthetic_context)
from geometry_lab.validate_jacobian import (load_settings,validate_case,run_validation,
    expanded_context,select_samples)


ROOT = Path(__file__).resolve().parents[1]
TOL = {"atol":1e-8,"rtol":1e-5,"signal_floor":1e-10}
torch.set_num_threads(2)


class JacobianTests(unittest.TestCase):
    def test_layout_and_analytic_polynomial(self):
        p = torch.tensor([0.7,1.3],dtype=torch.float64)
        fn = lambda v: torch.stack((v[0].square(),v[0]*v[1],v[1].exp())).reshape(1,1,3)
        j = jacobian_jvp(fn,p)
        expected = torch.tensor([[1.4,1.3,0],[0,0.7,float(p[1].exp())]],dtype=p.dtype).reshape(2,1,1,3)
        self.assertEqual(j.shape,(2,1,1,3))
        torch.testing.assert_close(j,expected)
        full_reverse = torch.autograd.functional.jacobian(fn,p)
        torch.testing.assert_close(j,full_reverse.movedim(-1,0))

    def test_fixed_scale_central_difference(self):
        p = torch.tensor([0.2,2.0],dtype=torch.float64)
        fn = lambda v: torch.stack((torch.sin(v[0]),v[1].pow(3)))
        fd = finite_difference_jacobian(fn,p,[1e-5,2e-5])
        self.assertEqual(fd.schemes,["central2","central2"])
        torch.testing.assert_close(fd.jacobian,jacobian_jvp(fn,p),rtol=1e-8,atol=1e-10)

    def test_central_difference_second_order_convergence(self):
        p = torch.tensor([0.7],dtype=torch.float64)
        fn = lambda v: v.exp()
        j = jacobian_jvp(fn,p)
        errors = [float((finite_difference_jacobian(fn,p,[h]).jacobian-j).abs().max()) for h in (0.08,0.04,0.02)]
        self.assertTrue(3.9 < errors[0]/errors[1] < 4.1)
        self.assertTrue(3.9 < errors[1]/errors[2] < 4.1)

    def test_second_order_boundary_stencils(self):
        fn = lambda v: v.square()
        left = torch.tensor([0.0],dtype=torch.float64)
        right = torch.tensor([1.0],dtype=torch.float64)
        a = finite_difference_jacobian(fn,left,[0.01],bounds=[(0,1)])
        b = finite_difference_jacobian(fn,right,[0.01],bounds=[(0,1)])
        self.assertEqual(a.schemes,["forward2"])
        self.assertEqual(b.schemes,["backward2"])
        torch.testing.assert_close(a.jacobian,torch.zeros((1,1),dtype=left.dtype),atol=1e-12,rtol=0)
        torch.testing.assert_close(b.jacobian,torch.ones((1,1),dtype=right.dtype)*2,atol=1e-12,rtol=0)

    def test_unavailable_stencil_is_not_zero(self):
        p = torch.tensor([0.0],dtype=torch.float64)
        r = finite_difference_jacobian(lambda v:v,p,[0.8],bounds=[(0,1)])
        self.assertEqual(r.schemes,["unavailable"])
        self.assertTrue(torch.isnan(r.jacobian).all())
        r = finite_difference_jacobian(lambda v:v,p,[0.1],bounds=[(0,1)],boundary="skip")
        self.assertEqual(r.schemes,["unavailable"])

    def test_float32_decimal_bound_and_unresolvable_step(self):
        p = torch.tensor([0.7],dtype=torch.float32)
        r = finite_difference_jacobian(lambda v:v,p,[0.01],bounds=[(0.7,1.0)])
        self.assertEqual(r.schemes,["forward2"])
        r = finite_difference_jacobian(lambda v:v,p,[1e-12])
        self.assertEqual(r.schemes,["unavailable"])

    def test_zero_derivative_uses_absolute_gate(self):
        a = torch.zeros((3,4,4),dtype=torch.float64)
        r = derivative_error(a,a+1e-12,**TOL)
        self.assertTrue(r["passed"])
        self.assertTrue(r["near_zero_reference"])
        self.assertIsNone(r["relative_l2"])
        self.assertIsNone(r["cosine"])
        self.assertFalse(derivative_error(a,a+1e-4,**TOL)["passed"])

    def test_nonfinite_errors_and_shape_mismatch(self):
        a = torch.ones((2,2),dtype=torch.float64)
        self.assertFalse(derivative_error(a,a*float("nan"),**TOL)["finite"])
        with self.assertRaises(ValueError):
            derivative_error(a,a.flatten(),**TOL)

    def test_adjacent_window_rejects_single_best_step(self):
        self.assertFalse(adjacent_pass_window([False,True,False,True])["passed"])
        r = adjacent_pass_window([False,True,True,False])
        self.assertEqual((r["start_index"],r["end_index"]),(1,2))
        with self.assertRaises(ValueError):
            adjacent_pass_window([True],1)

    def test_reverse_projection_independent_check(self):
        p = torch.tensor([0.7,1.3],dtype=torch.float64)
        fn = lambda v: torch.stack((v[0].square(),v[0]*v[1],v[1].exp()))
        j = jacobian_jvp(fn,p)
        self.assertTrue(all(r["passed"] for r in reverse_projection_check(fn,p,j)))
        self.assertFalse(all(r["passed"] for r in reverse_projection_check(fn,p,j*1.01)))

    def test_higher_order_graph_is_retained_only_on_request(self):
        p = torch.tensor([0.7],dtype=torch.float64,requires_grad=True)
        fn = lambda v: v.pow(3)
        j = jacobian_jvp(fn,p,create_graph=True)
        g, = torch.autograd.grad(j.sum(),p)
        torch.testing.assert_close(g,6*p)
        self.assertFalse(jacobian_jvp(fn,p).requires_grad)
        with self.assertRaises(ValueError):
            jacobian_jvp(fn,p.detach(),create_graph=True)

    def test_all_operator_closed_form_references(self):
        x = synthetic_context("textured",8,3)
        fields = auxiliary_fields(x,123)
        for spec in operator_specs([0.5,1.2]).values():
            for values in spec.points:
                with self.subTest(operator=spec.name,values=values):
                    p = torch.tensor(values,dtype=x.dtype)
                    fn = bind_operator(spec,x,halo=3,core_size=8,kernel_size=7,fields=fields)
                    analytic = analytic_jacobian(spec,x,p,halo=3,core_size=8,fields=fields,kernel_size=7)
                    self.assertIsNotNone(analytic)
                    torch.testing.assert_close(jacobian_jvp(fn,p),analytic,rtol=1e-10,atol=1e-12)

    def test_anisotropic_isotropy_zero_angle_and_width_sum(self):
        x = synthetic_context("textured",8,3)
        fields = auxiliary_fields(x,123)
        specs = operator_specs([0.5,1.2])
        fn_a = bind_operator(specs["anisotropic_blur"],x,halo=3,core_size=8,kernel_size=7,fields=fields)
        fn_i = bind_operator(specs["isotropic_blur"],x,halo=3,core_size=8,kernel_size=7,fields=fields)
        ja = jacobian_jvp(fn_a,torch.tensor([0.8,0.8,0.47],dtype=x.dtype))
        ji = jacobian_jvp(fn_i,torch.tensor([0.8],dtype=x.dtype))
        torch.testing.assert_close(ja[2],torch.zeros_like(ja[2]),atol=1e-12,rtol=0)
        torch.testing.assert_close(ja[0]+ja[1],ji[0],atol=1e-12,rtol=1e-10)

    def test_blur_jacobian_retains_derivatives_for_later_geometry(self):
        x = synthetic_context("textured",6,3)
        spec = operator_specs([0.5,1.2])["isotropic_blur"]
        fn = bind_operator(spec,x,halo=3,core_size=6,kernel_size=7,fields=auxiliary_fields(x,123))
        p = torch.tensor([0.8],dtype=x.dtype,requires_grad=True)
        j = jacobian_jvp(fn,p,create_graph=True)
        energy = j.square().mean()
        gradient, = torch.autograd.grad(energy,p,create_graph=True)
        second, = torch.autograd.grad(gradient.sum(),p)
        h = 1e-5
        plus = jacobian_jvp(fn,p.detach()+h).square().mean()
        minus = jacobian_jvp(fn,p.detach()-h).square().mean()
        torch.testing.assert_close(gradient,(plus-minus).reshape(1)/(2*h),atol=1e-8,rtol=1e-6)
        self.assertTrue(torch.isfinite(second).all())

    def test_buggy_detachment_is_caught_by_fd(self):
        x = torch.arange(12,dtype=torch.float64).reshape(3,2,2)/12
        fn = lambda p: p.detach()[0]*x
        p = torch.tensor([1.0],dtype=x.dtype)
        settings = load_settings(ROOT/"configs/jacobian.yaml")
        spec = copy.deepcopy(operator_specs([0.5,1.2])["isotropic_blur"])
        spec.bounds = ((0.5,1.5),)
        result = validate_case(fn,p,spec,settings)
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"][0]["passed"])

    def test_stochastic_resampling_is_caught(self):
        generator = torch.Generator().manual_seed(11)
        fn = lambda p: p[0]+torch.randn((3,2,2),generator=generator,dtype=p.dtype)
        p = torch.tensor([1.0],dtype=torch.float64)
        settings = load_settings(ROOT/"configs/jacobian.yaml")
        spec = operator_specs([0.5,1.5])["isotropic_blur"]
        result = validate_case(fn,p,spec,settings)
        self.assertFalse(result["deterministic"])
        self.assertFalse(result["passed"])

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            jacobian_jvp(lambda p:p,torch.ones(2,2))
        with self.assertRaises(ValueError):
            finite_difference_jacobian(lambda p:p,torch.ones(1,dtype=torch.float64),[0])
        with self.assertRaises(ValueError):
            finite_difference_jacobian(lambda p:p,torch.ones(1,dtype=torch.float64),[0.1],bounds=[(0,0.5)])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_config(ROOT/"configs/default.yaml")
        self.settings = load_settings(ROOT/"configs/jacobian.yaml")
        self.settings.update(plots=False,save_example_tensors=False,control_patterns=["textured"],noise_realizations=1,
                             operators=["affine_intensity","gaussian_noise","isotropic_blur","anisotropic_blur","haze_toy_depth"],
                             relative_steps=[0.003,0.001,0.0003,0.0001],reverse_probes=1,control_core_size=6,output_dir=str(self.root/"reports"))
        self.cfg["sampling"].update(crop_size=8,halo=3,crops_per_image=1)
        self.cfg["operators"].update(kernel_size=7,sigma_domain=[0.5,0.7])

    def tearDown(self):
        self.temp.cleanup()

    def make_manifests(self):
        sampling = self.cfg["sampling"]
        output = self.root/"task01"
        output.mkdir()
        for split,seed in (("train",13),("valid",17)):
            folder = self.root/split
            folder.mkdir()
            # Large margins leave room for support-study context.
            array = np.random.default_rng(seed).integers(0,256,(80,80,3),dtype=np.uint8)
            Image.fromarray(array).save(folder/"0001.png")
            manifest,_ = build_split_manifest(folder,split,sampling,1)
            # Use an explicit eligible fixed crop for integration tests only.
            manifest["records"][0].update(top=30,left=30)
            (output/f"{split}_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            self.cfg["dataset"]["train_hr" if split == "train" else "valid_hr"] = str(folder)
        self.cfg["experiment"]["output_dir"] = str(output)

    def test_control_runner_writes_reports_and_refuses_existing_run(self):
        summary,dest = run_validation(self.cfg,self.settings,stage="controls")
        self.assertEqual(summary["status"],"PASS")
        self.assertTrue((dest/"finite_difference_sweep.csv").exists())
        self.assertTrue((dest/"analytic_checks.csv").exists())
        self.assertTrue((dest/"run_metadata.json").exists())
        with self.assertRaises(FileExistsError):
            run_validation(self.cfg,self.settings,stage="controls")

    def test_pilot_precision_and_support_integration(self):
        self.make_manifests()
        for stage in ("pilot","precision","support"):
            self.settings[stage].update(max_train_images=1,max_valid_images=0,crops_per_image=1)
        self.settings["support"]["larger_kernel_size"] = 9
        self.settings["operators"] = ["affine_intensity","gaussian_noise","isotropic_blur"]
        pilot,_ = run_validation(self.cfg,self.settings,stage="pilot")
        precision,_ = run_validation(self.cfg,self.settings,stage="precision")
        support,_ = run_validation(self.cfg,self.settings,stage="support")
        self.assertEqual(pilot["status"],"PASS")
        self.assertEqual(precision["status"],"PASS")
        self.assertEqual(support["status"],"PASS")
        self.assertGreater(support["comparison_rows"],0)

    def test_support_never_synthesizes_missing_physical_context(self):
        self.make_manifests()
        sample = {"record":{"file_name":"0001.png","top":3,"left":3,"crop_size":8},"root":str(self.root/"train")}
        self.assertIsNone(expanded_context(sample,8,"srgb"))
        self.settings["support"].update(max_train_images=1,max_valid_images=0,larger_kernel_size=9)
        self.settings["operators"] = ["isotropic_blur"]
        path = self.root/"task01"/"train_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["records"][0].update(top=3,left=3)
        path.write_text(json.dumps(manifest))
        summary,_ = run_validation(self.cfg,self.settings,stage="support")
        self.assertEqual(summary["status"],"INCOMPLETE")
        self.assertGreater(summary["unavailable_support_entries"],0)

    def test_explicit_image_selection_and_out_of_range_rejection(self):
        self.make_manifests()
        self.settings["pilot"].update(train_image_indices=[0],valid_image_indices=[],max_train_images=4,max_valid_images=2)
        samples,metadata = select_samples(self.cfg,self.settings,"pilot")
        self.assertEqual(len(samples),1)
        self.assertEqual(metadata["train"]["selected_image_indices"],[0])
        self.assertNotIn("valid",metadata)
        self.settings["pilot"]["train_image_indices"] = [1]
        with self.assertRaises(ValueError):
            select_samples(self.cfg,self.settings,"pilot")


if __name__ == "__main__":
    unittest.main()
