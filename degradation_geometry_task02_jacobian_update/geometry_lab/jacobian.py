"""Small-parameter, large-output Jacobians with an explicit parameter-first layout.

For F: R^k -> R^(C H W), returned shape is (k,C,H,W), NOT (C,H,W,k).
Forward-mode JVPs need k evaluations, avoiding one reverse pass per image pixel.
All differences are computed on raw floating tensors, before clipping/quantizing.
"""
from dataclasses import dataclass
import math
from typing import Callable

import torch


@dataclass
class FiniteDifferenceResult:
    jacobian: torch.Tensor
    steps: list[float]
    schemes: list[str]


def _check_point(xi):
    if not isinstance(xi, torch.Tensor) or xi.ndim != 1 or not xi.is_floating_point():
        raise ValueError("xi must be a one-dimensional floating tensor.")
    if xi.dtype not in (torch.float32, torch.float64) or xi.numel() < 1:
        raise ValueError("Use a nonempty float32 or float64 parameter vector.")
    if not torch.isfinite(xi).all():
        raise ValueError("Parameter point contains nonfinite values.")


def _check_output(value, xi):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError("Operator must return one floating tensor.")
    if value.dtype != xi.dtype or value.device != xi.device:
        raise ValueError("Operator output must have xi's dtype/device.")
    if value.numel() < 1 or not torch.isfinite(value).all():
        raise ValueError("Operator produced empty/nonfinite output.")


def jacobian_jvp(fn: Callable, xi: torch.Tensor, *, create_graph=False):
    """J[i] = dF/dxi_i via forward AD, retaining graph only on request.

    create_graph=True requires a requires_grad parameter tensor supplied by the
    caller. It preserves differentiation through J for future metric derivatives.
    Constant/zero columns legitimately may not themselves require gradients.
    """
    _check_point(xi)
    if create_graph and not xi.requires_grad:
        raise ValueError("For create_graph=True, supply xi.requires_grad=True.")
    columns = []
    basis = torch.eye(xi.numel(), dtype=xi.dtype, device=xi.device)
    with torch.enable_grad(), torch.autocast(device_type=xi.device.type, enabled=False):
        for tangent in basis:
            value, derivative = torch.func.jvp(fn, (xi,), (tangent,), strict=False)
            _check_output(value, xi)
            _check_output(derivative, xi)
            columns.append(derivative if create_graph else derivative.detach())
    return torch.stack(columns, dim=0)


def finite_difference_jacobian(fn: Callable, xi: torch.Tensor, steps,
                               *, bounds=None, boundary="one_sided"):
    """Central O(h^2) differences; O(h^2) one-sided differences at box edges.

    steps are ABSOLUTE per-coordinate steps, e.g. epsilon*[1 pixel,1 radian].
    No step is silently shrunk. A step that cannot fit a second-order stencil
    is labeled 'unavailable' and its column is filled with NaNs. Do not treat
    that column as a zero derivative. bounds use closed study-domain intervals.
    """
    _check_point(xi)
    if boundary not in ("one_sided", "skip"):
        raise ValueError("boundary must be one_sided or skip.")
    hvalues = torch.as_tensor(steps, dtype=xi.dtype, device=xi.device)
    if hvalues.shape != xi.shape or not torch.isfinite(hvalues).all() or (hvalues <= 0).any():
        raise ValueError("steps must be finite, positive and have xi.shape.")
    box = [(None, None)] * xi.numel() if bounds is None else list(bounds)
    if len(box) != xi.numel():
        raise ValueError("bounds must have one interval per coordinate.")
    # Compare against bounds represented in the SAME dtype as xi. A decimal
    # endpoint such as 0.7 may round downward in float32, without being outside
    # its declared chart. This conversion does not change/stabilize the stencil.
    box = [(None if lo is None else float(torch.as_tensor(lo,dtype=xi.dtype)),
            None if hi is None else float(torch.as_tensor(hi,dtype=xi.dtype))) for lo,hi in box]
    for value, (lo, hi) in zip(xi.detach().cpu().tolist(), box):
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError("Parameter point lies outside bounds.")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("Bounds require lo < hi.")
    columns, schemes, actual_steps = [], [], []
    with torch.no_grad(), torch.autocast(device_type=xi.device.type, enabled=False):
        base = fn(xi)
        _check_output(base, xi)
        for i, (lo, hi) in enumerate(box):
            center = float(xi[i])
            step = float(hvalues[i])
            delta = torch.zeros_like(xi)
            delta[i] = hvalues[i]
            # Float32 perturbations may round to xi: expose this explicitly.
            distinguishable = bool((xi[i]+hvalues[i] != xi[i]) and (xi[i]-hvalues[i] != xi[i]))
            inside = lambda v: (lo is None or v >= lo) and (hi is None or v <= hi)
            if distinguishable and inside(center-step) and inside(center+step):
                a, b = fn(xi+delta), fn(xi-delta)
                _check_output(a, xi)
                _check_output(b, xi)
                derivative = (a-b)/(2*hvalues[i])
                scheme = "central2"
            elif distinguishable and boundary == "one_sided" and inside(center+2*step):
                a, b = fn(xi+delta), fn(xi+2*delta)
                _check_output(a, xi)
                _check_output(b, xi)
                derivative = (-3*base+4*a-b)/(2*hvalues[i])
                scheme = "forward2"
            elif distinguishable and boundary == "one_sided" and inside(center-2*step):
                a, b = fn(xi-delta), fn(xi-2*delta)
                _check_output(a, xi)
                _check_output(b, xi)
                derivative = (3*base-4*a+b)/(2*hvalues[i])
                scheme = "backward2"
            else:
                derivative = torch.full_like(base, float("nan"))
                scheme = "unavailable"
            columns.append(derivative)
            schemes.append(scheme)
            actual_steps.append(step)
    return FiniteDifferenceResult(torch.stack(columns), actual_steps, schemes)


def derivative_error(reference, candidate, *, atol, rtol, signal_floor):
    """Mixed absolute/relative RMS gate, safe for true zero derivatives.

    The denominator is NOT floored to force a relative-error number. Relative
    error and cosine are null for unresolved reference directions; use abs RMS.
    All reductions use float64 so float32 diagnostics do not add reduction error.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Derivative shapes differ.")
    a, b = reference.detach().double().flatten(), candidate.detach().double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"reference_rms": None, "candidate_rms": None, "error_rms": None,
                "max_abs_error": None, "relative_l2": None, "cosine": None,
                "near_zero_reference": False, "passed": False, "finite": False}
    ar = float(a.square().mean().sqrt())
    br = float(b.square().mean().sqrt())
    err = float((a-b).square().mean().sqrt())
    near_zero = ar <= signal_floor
    cosine = None
    if not near_zero and br > signal_floor:
        cosine = float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)))
    return {"reference_rms": ar, "candidate_rms": br, "error_rms": err,
            "max_abs_error": float((a-b).abs().max()),
            "relative_l2": None if near_zero else err/ar, "cosine": cosine,
            "near_zero_reference": near_zero, "passed": err <= atol+rtol*ar,
            "finite": True}


def adjacent_pass_window(passes, minimum=2):
    """Require a stable step interval, not a cherry-picked best epsilon."""
    if minimum < 2:
        raise ValueError("At least two adjacent passing steps are required.")
    run = 0
    for idx, passed in enumerate(passes):
        run = run+1 if passed else 0
        if run >= minimum:
            return {"passed": True, "start_index": idx-minimum+1, "end_index": idx}
    return {"passed": False, "start_index": None, "end_index": None}


def reverse_projection_check(fn, xi, jacobian, *, seed=161803, probes=3, tolerances=None):
    """Independent reverse-mode check of random linear output projections.

    Compare grad_xi <w,F(xi)> with [<w,J_0>,...,<w,J_(k-1)>]. This costs a
    few reverse passes, rather than constructing a pixel-by-pixel reverse J.
    Projection equality supplements, and never replaces, full-image FD checks.
    """
    _check_point(xi)
    if probes < 1:
        raise ValueError("probes must be positive.")
    tolerance = tolerances or {"atol":1e-10, "rtol":1e-8, "signal_floor":1e-12}
    p = xi.detach().clone().requires_grad_(True)
    results = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for probe in range(probes):
        with torch.enable_grad(), torch.autocast(device_type=xi.device.type, enabled=False):
            y = fn(p)
            _check_output(y, p)
            w = torch.randn(y.shape, generator=generator, dtype=torch.float64).to(y.device, y.dtype)
            w = w / math.sqrt(y.numel())
            projected = (y*w).sum()
            if projected.requires_grad:
                reverse, = torch.autograd.grad(projected, p, allow_unused=True)
                reverse = torch.zeros_like(p) if reverse is None else reverse
            else:
                reverse = torch.zeros_like(p)
            forward = (jacobian*w).reshape(p.numel(),-1).sum(1)
        for i in range(p.numel()):
            results.append({"probe":probe, "coordinate":i,
                            **derivative_error(forward[i],reverse[i],**tolerance)})
    return results
