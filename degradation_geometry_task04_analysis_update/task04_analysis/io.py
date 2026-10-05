"""Strict Task 3 imports and atomic Task 4 exports. No source-image access."""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_settings(path):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    keys = {"plots", "rank_atol", "rank_rtol", "psd_atol", "psd_rtol", "eigengap_atol",
            "eigengap_rtol", "rank_sweep_atol", "rank_sweep_rtol", "reference_point"}
    if not isinstance(cfg, dict) or set(cfg) != keys:
        raise ValueError(f"Task 4 config must have exactly these keys: {sorted(keys)}")
    if type(cfg["plots"]) is not bool or cfg["reference_point"] != "median":
        raise ValueError("plots must be boolean; reference_point must be median.")
    for name in ("rank_atol", "rank_rtol", "psd_atol", "psd_rtol", "eigengap_atol", "eigengap_rtol"):
        v = cfg[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"Invalid {name}: require a finite nonnegative number.")
    for name in ("rank_sweep_atol", "rank_sweep_rtol"):
        values = cfg[name]
        if (not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not np.isfinite(v) or v < 0 for v in values) or len(set(values)) != len(values)):
            raise ValueError(f"Invalid {name}: require unique finite nonnegative values.")
    return cfg


def load_run(path):
    root = Path(path).resolve()
    run = read_json(root/"run.json")
    if run.get("schema_version") != 1 or fingerprint(run["plan"]) != run.get("fingerprint"):
        raise ValueError("Invalid Task 3 run metadata/fingerprint.")
    return root, run


def safe_name(name):
    return isinstance(name, str) and bool(name) and name not in (".", "..") and all(c.isalnum() or c == "_" for c in name)


def load_matrices(root, run, operator, split):
    plan = run["plan"]
    if plan["stage"] == "controls":
        raise ValueError("Controls use controls.json. Supply a pilot/full/synthetic image run for metric.npz.")
    if not safe_name(operator) or split not in ("train", "valid"):
        raise ValueError("Unsafe operator name or invalid split.")
    grid = plan["grids"][operator]
    selected = plan["selections"][split]["selected"]
    path = Path(root)/"analysis"/operator/split/"metric.npz"
    if not path.is_file():
        raise ValueError(f"Missing aggregate matrix: {path}. Use a completed pilot or operator/split aggregate.")
    with np.load(path, allow_pickle=False) as data:
        required = {"points", "scales", "mean_metric", "per_image_metric", "image_ids", "crop_counts"}
        if not required <= set(data.files):
            raise ValueError(f"Missing matrix arrays: {sorted(required-set(data.files))}")
        arrays = {name: data[name].copy() for name in required}
    k, p, n = len(grid["scales"]), len(grid["points"]), len(selected)
    mean, images = arrays["mean_metric"], arrays["per_image_metric"]
    if (mean.dtype != np.float64 or images.dtype != np.float64 or mean.shape != (p, k, k)
            or images.shape != (n, p, k, k) or not n or not np.isfinite(mean).all() or not np.isfinite(images).all()):
        raise ValueError("Invalid matrix dtype, shape or nonfinite entries.")
    if not np.array_equal(arrays["points"], np.asarray(grid["points"])) or not np.array_equal(arrays["scales"], grid["scales"]):
        raise ValueError("Matrix points/scales differ from run metadata.")
    ids = [item["image"]["image_id"] for item in selected]
    counts = [len(item["records"]) for item in selected]
    if (arrays["image_ids"].tolist() != ids or arrays["crop_counts"].tolist() != counts
            or len(set(ids)) != n or any(c < 1 for c in counts)):
        raise ValueError("Matrix image IDs or crop counts differ from run metadata.")
    if not np.allclose(mean, images.mean(0), rtol=1e-12, atol=1e-15):
        raise ValueError("Stored mean is not the equal-image average.")
    if len(grid["parameter_names"]) != k or len(set(grid["parameter_names"])) != k:
        raise ValueError("Invalid coordinate names.")
    if grid["axes"] is not None:
        axes = grid["axes"]
        if (len(axes) != k or any(not len(a) or np.any(np.diff(a) <= 0) for a in axes)
                or not np.array_equal(np.asarray(list(itertools.product(*axes))), arrays["points"])):
            raise ValueError("Cartesian axis metadata does not match points/order.")
    return {"grid": grid, "arrays": arrays, "source_path": path, "source_sha256": sha256(path)}


def load_controls(path):
    root, run = load_run(path)
    if run["plan"]["stage"] != "controls":
        raise ValueError("--controls-run must identify a Task 3 controls run.")
    data = read_json(root/"controls.json")
    rows = data.get("rows", [])
    if data.get("stage") != "controls" or data.get("cases") != len(rows) or not rows:
        raise ValueError("Invalid/empty controls payload.")
    failed = sum(not r["passed"] for r in rows)
    if data.get("failed") != failed or failed or any(not all(r["checks"].values()) for r in rows):
        raise ValueError("Task 3 controls are inconsistent or failed; resolve them before Task 4.")
    groups = {}
    for row in rows:
        key = (row["operator"], row["pattern"])
        groups.setdefault(key, []).append(row)
    return root, run, groups, sha256(root/"controls.json")
