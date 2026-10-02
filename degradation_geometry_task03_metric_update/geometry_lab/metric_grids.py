"""Explicit bounded parameter grids; no hidden interpolation or adaptive fitting."""
import itertools
import numpy as np
import yaml
from .jacobian_cases import operator_specs


def load_metric_config(path, sigma_domain):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Metric configuration must be a mapping.")
    for key in ("torch_threads", "noise_realizations"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(cfg.get("noise_seed")) is not int:
        raise ValueError("noise_seed must be an integer.")
    if not isinstance(cfg.get("output_dir"), str) or not cfg["output_dir"]:
        raise ValueError("output_dir must be a nonempty path string.")
    if type(cfg.get("plots")) is not bool:
        raise ValueError("plots must be a boolean.")
    if not isinstance(cfg.get("device"), str):
        raise ValueError("device must be a torch device string.")
    for key in ("max_train_images", "max_valid_images", "crops_per_image"):
        if type(cfg["pilot"].get(key)) is not int or cfg["pilot"][key] < 1:
            raise ValueError(f"pilot.{key} must be a positive integer.")
    for key in ("rank_atol", "rank_rtol", "psd_atol", "psd_rtol"):
        value = cfg["diagnostics"].get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid diagnostics.{key}.")
    specs = operator_specs(sigma_domain)
    if not isinstance(cfg.get("grids"), dict) or not cfg["grids"]:
        raise ValueError("Define at least one operator grid.")
    grids = {}
    for name, grid in cfg["grids"].items():
        if name not in specs:
            raise ValueError(f"Unknown verified operator: {name}")
        spec = specs[name]
        if not isinstance(grid, dict) or set(grid) not in ({"axes"}, {"points"}):
            raise ValueError(f"{name}: supply exactly axes or points.")
        axes = None
        if "axes" in grid:
            if set(grid["axes"]) != set(spec.parameter_names):
                raise ValueError(f"{name}: axes must be {spec.parameter_names}.")
            axes = []
            for parameter in spec.parameter_names:
                values = np.asarray(grid["axes"][parameter], dtype=np.float64)
                if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
                    raise ValueError(f"{name}/{parameter}: levels must be finite and strictly increasing.")
                axes.append(values.tolist())
            points = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
        else:
            points = np.asarray(grid["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != len(spec.parameter_names) or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"{name}: invalid parameter points.")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError(f"{name}: duplicate parameter points.")
        for j, (lo, hi) in enumerate(spec.bounds):
            if np.any(points[:, j] < lo) or np.any(points[:, j] > hi):
                raise ValueError(f"{name}: {spec.parameter_names[j]} outside [{lo}, {hi}].")
        grids[name] = {"points": points.tolist(), "axes": axes,
                       "parameter_names": list(spec.parameter_names), "scales": list(spec.scales),
                       "bounds": [list(b) for b in spec.bounds],
                       "fixed_theta_rad": 0.31 if name == "anisotropic_fixed_theta" else None}
    return cfg, grids, specs
