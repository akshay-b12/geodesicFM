"""Configuration validation and paths, relative to the project root."""
from pathlib import Path
import yaml


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    root = config_path.parent.parent
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(root)
    s = cfg["sampling"]
    for key in ("crop_size", "crops_per_image"):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"sampling.{key} must be a positive integer.")
    if s["crop_size"] < 2:
        raise ValueError("crop_size must be >= 2 for adjacent-pixel content statistics.")
    if type(s["halo"]) is not int or s["halo"] < 0:
        raise ValueError("sampling.halo must be a nonnegative integer.")
    if type(s["seed"]) is not int:
        raise ValueError("sampling.seed must be an integer.")
    if s["color_space"] not in ("srgb", "linear_rgb"):
        raise ValueError("color_space must be srgb or linear_rgb.")
    k = cfg["operators"]["kernel_size"]
    if type(k) is not int or k < 3 or k % 2 != 1:
        raise ValueError("kernel_size must be an odd integer >= 3.")
    lo, hi = cfg["operators"]["sigma_domain"]
    if not 0 < lo <= hi:
        raise ValueError("Require 0 < sigma_min <= sigma_max.")
    radius = k // 2
    if radius > s["halo"]:
        raise ValueError("halo must be >= kernel radius for boundary-free core measurements.")
    if radius < cfg["operators"]["support_sigma_multiple"] * hi:
        raise ValueError("Fixed kernel support is too small for sigma_domain.")
    if cfg["experiment"]["dtype"] not in ("float32", "float64"):
        raise ValueError("Use float32 or float64; float64 is recommended for geometry.")
    if cfg["preview"]["split"] != "train":
        raise ValueError("Exploratory previews must use the training split.")
    return cfg


def resolve_path(cfg, value):
    if not value:
        raise ValueError("Set dataset.train_hr and dataset.valid_hr in configs/default.yaml.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(cfg["_project_root"]) / path).resolve()


def output_path(cfg):
    return resolve_path(cfg, cfg["experiment"]["output_dir"])
