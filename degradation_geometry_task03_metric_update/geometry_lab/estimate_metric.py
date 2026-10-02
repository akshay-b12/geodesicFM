"""Image-averaged metric estimation with per-image atomic checkpoints.

Run: python -m geometry_lab.estimate_metric --stage pilot --config ...
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from .config import load_config, resolve_path, output_path
from .datasets import decode_rgb, rgb_tensor, file_sha256, stable_seed, core_crop
from .jacobian import jacobian_jvp
from .jacobian_cases import (auxiliary_fields, bind_operator, analytic_jacobian,
                             synthetic_context)
from .metric import gram_metric, metric_diagnostics
from .metric_grids import load_metric_config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initialize_run(directory, plan, resume=False):
    directory = Path(directory)
    identity = fingerprint(plan)
    marker = directory / "run.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError("Run already exists. Use --resume or a new --run-name.")
        if previous.get("fingerprint") != identity or fingerprint(previous["plan"]) != identity:
            raise ValueError("Resume fingerprint differs: data, code, grid, settings or runtime changed. Use a new run name.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Refusing a nonempty directory without run.json.")
        atomic_json(marker, {"schema_version": 1, "fingerprint": identity, "plan": plan})
    return identity


def shard_path(directory, name, split, image_id):
    # Image names never become paths; safe on Windows as well as Linux.
    return Path(directory)/"checkpoints"/name/split/(fingerprint(image_id)[:24]+".npz")


def read_shard(path, identity, image_id, shape):
    path = Path(path)
    marker = path.with_suffix(".json")
    if not marker.exists():
        return None  # An uncommitted/partially written NPZ is recomputed.
    info = json.loads(marker.read_text(encoding="utf-8"))
    if info.get("fingerprint") != identity or info.get("image_id") != image_id:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    if not path.exists() or file_sha256(path) != info.get("sha256"):
        raise ValueError(f"Damaged checkpoint: {path}. Remove this NPZ and its JSON marker, then resume.")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"image_metric", "crop_metrics", "realization_trace_std", "crop_indices"}:
            raise ValueError(f"Invalid checkpoint fields: {path}")
        result = {key: data[key].copy() for key in data.files}
    crops = len(result["crop_indices"])
    if (result["image_metric"].shape != shape or result["crop_metrics"].shape != (crops, *shape)
            or result["realization_trace_std"].shape != (crops, shape[0])
            or not all(np.isfinite(a).all() for a in result.values())):
        raise ValueError(f"Invalid checkpoint arrays: {path}")
    if not np.array_equal(result["image_metric"], result["crop_metrics"].mean(0)):
        raise ValueError(f"Checkpoint image/crop average mismatch: {path}")
    return result


def write_shard(path, identity, image_id, arrays):
    atomic_npz(path, **arrays)
    # Marker is the commit point and includes a content checksum.
    atomic_json(Path(path).with_suffix(".json"), {
        "fingerprint": identity, "image_id": image_id, "sha256": file_sha256(path)})


def select_manifest(cfg, split, stage, settings):
    path = output_path(cfg)/f"{split}_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("split") != split:
        raise ValueError(f"Wrong manifest schema/split: {path}")
    if manifest["sampling"] != cfg["sampling"]:
        raise ValueError("Sampling config differs from prepared manifest; restore it or prepare again.")
    images = manifest["images"]
    if not images or len({i["image_id"] for i in images}) != len(images):
        raise ValueError("Manifest must contain unique, nonempty image IDs.")
    expected = cfg["dataset"].get(f"expected_{split}_images")
    if expected is not None and len(images) != expected:
        raise ValueError(f"{split}: manifest count differs from configured expected image count.")
    if stage == "pilot":
        count = min(len(images), settings["pilot"][f"max_{split}_images"])
        indices = np.linspace(0, len(images)-1, count, dtype=int)
        images = [images[i] for i in indices]
    root = resolve_path(cfg, cfg["dataset"][f"{split}_hr"])
    selected = []
    for info in images:
        source = root/info["file_name"]
        if file_sha256(source) != info["sha256"]:
            raise ValueError(f"Image changed since prepare: {source}")
        records = sorted((r for r in manifest["records"] if r["image_id"] == info["image_id"]),
                         key=lambda r: r["crop_index"])
        if (len(records) != cfg["sampling"]["crops_per_image"]
                or [r["crop_index"] for r in records] != list(range(len(records)))):
            raise ValueError("Missing or duplicate crop records.")
        for r in records:
            if (r["file_name"] != info["file_name"] or r["crop_size"] != cfg["sampling"]["crop_size"]
                    or r["halo"] != cfg["sampling"]["halo"]
                    or not r["halo"] <= r["top"] <= info["height"]-r["crop_size"]-r["halo"]
                    or not r["halo"] <= r["left"] <= info["width"]-r["crop_size"]-r["halo"]):
                raise ValueError("Invalid physical-context crop record.")
        if stage == "pilot":
            records = records[:settings["pilot"]["crops_per_image"]]
        selected.append({"image": info, "records": records})
    return {"manifest_sha256": file_sha256(path), "prepared_images": len(manifest["images"]),
            "selected": selected}, root


def estimate_image(spec, points, array, records, sampling, settings, kernel_size, device):
    crop_metrics, trace_std = [], []
    for rec in records:
        t, l, h, size = (rec[k] for k in ("top", "left", "halo", "crop_size"))
        patch = array[t-h:t+size+h, l-h:l+size+h]
        if patch.shape[:2] != (size+2*h, size+2*h):
            raise ValueError("Crop context is outside image.")
        x = rgb_tensor(patch, sampling["color_space"], torch.float64).to(device)
        repetitions = settings["noise_realizations"] if spec.has_noise else 1
        # Seeds deliberately omit parameters and operator name. CRN across
        # points and compatible noise operators, with independent image/crop draws.
        realized = []
        for realization in range(repetitions):
            seed = stable_seed("task03_noise", settings["noise_seed"], rec["image_id"],
                               rec["crop_index"], realization)
            fields = auxiliary_fields(x, seed)
            fn = bind_operator(spec, x, halo=h, core_size=size, kernel_size=kernel_size, fields=fields)
            values = []
            for point in points:
                p = torch.tensor(point, dtype=torch.float64, device=device)
                values.append(gram_metric(jacobian_jvp(fn, p)).cpu().numpy())
            realized.append(np.stack(values))
        realized = np.stack(realized)
        crop_metrics.append(realized.mean(0))
        traces = np.trace(realized, axis1=-2, axis2=-1)
        # Descriptive realization spread; this is NOT a standard error.
        trace_std.append(traces.std(0, ddof=1) if repetitions > 1 else np.zeros(len(points)))
    crop_metrics = np.stack(crop_metrics)
    return {"image_metric": crop_metrics.mean(0), "crop_metrics": crop_metrics,
            "realization_trace_std": np.stack(trace_std),
            "crop_indices": np.asarray([r["crop_index"] for r in records], dtype=np.int64)}


def run_controls(directory, settings, grids, specs, kernel_size, halo, device):
    rows = []
    # All 11 verified families, including charts with known exact null directions.
    for name, spec in specs.items():
        points = grids[name]["points"] if name in grids else spec.points
        for pattern in ("textured", "constant", "impulse"):
            x = synthetic_context(pattern, 16, halo).to(device)
            fields = auxiliary_fields(x, settings["noise_seed"])
            fn = bind_operator(spec, x, halo=halo, core_size=16, kernel_size=kernel_size, fields=fields)
            for point in points:
                p = torch.tensor(point, dtype=torch.float64, device=device)
                g = gram_metric(jacobian_jvp(fn, p)).cpu().numpy()
                reference = gram_metric(analytic_jacobian(spec, x, p, halo=halo, core_size=16,
                                                          fields=fields, kernel_size=kernel_size)).cpu().numpy()
                error = float(np.max(np.abs(g-reference)))
                limit = 1e-12+1e-10*float(np.max(np.abs(reference)))
                passed = error <= limit
                checks = {"analytic_gram": passed}
                if name == "affine_intensity":
                    core = core_crop(x, halo, 16)
                    closed = np.array([[float(core.square().mean()), float(core.mean())],
                                       [float(core.mean()), 1.0]])
                    checks["closed_form_affine"] = bool(np.allclose(g, closed, atol=1e-12, rtol=1e-10))
                if pattern == "constant" and name in ("isotropic_blur", "anisotropic_fixed_theta", "anisotropic_blur"):
                    checks["constant_blur_zero"] = bool(np.max(np.abs(g)) <= 1e-24)
                if name == "anisotropic_blur" and point[0] == point[1]:
                    checks["isotropic_theta_zero"] = bool(np.max(np.abs(g[2])) <= 1e-24)
                if name.startswith("haze_") and point[0] == 0:
                    checks["zero_beta_airlight_zero"] = bool(np.max(np.abs(g[1])) <= 1e-24)
                diagnostic = metric_diagnostics(g, spec.scales, **settings["diagnostics"])
                rows.append({"operator": name, "pattern": pattern, "point": point,
                             "max_abs_gram_error": error, "tolerance": limit, "checks": checks,
                             "passed": all(checks.values()), "metric": g.tolist(),
                             "diagnostics": diagnostic})
    summary = {"stage": "controls", "cases": len(rows),
               "failed": sum(not r["passed"] for r in rows), "rows": rows}
    atomic_json(Path(directory)/"controls.json", summary)
    print(f"Controls: {len(rows)} cases, {summary['failed']} failures.", flush=True)
    if summary["failed"]:
        raise RuntimeError("Metric analytic controls failed; see controls.json.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--metric-config", default="configs/metric.yaml")
    parser.add_argument("--stage", choices=("controls", "pilot", "full"), required=True)
    parser.add_argument("--split", choices=("train", "valid", "both"), default="both")
    parser.add_argument("--operators", nargs="+")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings, grids, specs = load_metric_config(args.metric_config, cfg["operators"]["sigma_domain"])
    if args.operators:
        if len(set(args.operators)) != len(args.operators) or any(n not in grids for n in args.operators):
            parser.error("--operators must be unique names configured in metric.yaml grids.")
        grids = {n: grids[n] for n in args.operators}
    torch.set_num_threads(settings["torch_threads"])
    device = torch.device(settings["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA float64 estimation is supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    # Geometry must not use reduced-precision convolution/matmul modes.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    splits = ("train", "valid") if args.split == "both" else (args.split,)
    selections, roots = {}, {}
    if args.stage != "controls":
        for split in splits:
            selections[split], roots[split] = select_manifest(cfg, split, args.stage, settings)
        if len(selections) == 2:
            for key in ("sha256", "pixel_sha256"):
                train = {r["image"][key] for r in selections["train"]["selected"]}
                valid = {r["image"][key] for r in selections["valid"]["selected"]}
                if train & valid:
                    raise ValueError("Selected train/validation images overlap.")
    code_hashes = {str(p.relative_to(Path(__file__).parent)): file_sha256(p)
                   for p in sorted(Path(__file__).parent.rglob("*.py"))}
    plan = {"stage": args.stage, "dtype": "float64", "sampling": cfg["sampling"],
            "operator_settings": cfg["operators"], "settings": settings, "grids": grids,
            "selections": selections, "code_hashes": code_hashes,
            "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                        "python": platform.python_version(), "device": str(device),
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()},
            "averaging": "output scalars -> realization Gram means -> crop means -> equal image means",
            "noise_seed_rule": "stable_seed('task03_noise', noise_seed, image_id, crop_index, realization)"}
    work = []
    for name, grid in grids.items():
        for split, selection in selections.items():
            crops = sum(len(r["records"]) for r in selection["selected"])
            repetitions = settings["noise_realizations"] if specs[name].has_noise else 1
            evaluations = crops*len(grid["points"])*repetitions
            work.append({"operator": name, "split": split, "images": len(selection["selected"]),
                         "crops": crops, "points": len(grid["points"]), "realizations": repetitions,
                         "jacobian_evaluations": evaluations,
                         "directional_jvps": evaluations*len(grid["parameter_names"])})
    print(json.dumps({"stage": args.stage, "work": work,
                      "fingerprint": fingerprint(plan)}, indent=2), flush=True)
    if args.dry_run:
        return 0
    run_name = args.run_name or args.stage
    if not run_name or run_name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Use a simple run name containing letters, digits, '_' or '-'.")
    directory = resolve_path(cfg, settings["output_dir"])/run_name
    identity = initialize_run(directory, plan, args.resume)
    if args.stage == "controls":
        run_controls(directory, settings, grids, specs, cfg["operators"]["kernel_size"], cfg["sampling"]["halo"], device)
    else:
        for split, selection in selections.items():
            for index, selected in enumerate(selection["selected"]):
                info, records = selected["image"], selected["records"]
                array = None
                for name, grid in grids.items():
                    shape = (len(grid["points"]), len(specs[name].parameter_names), len(specs[name].parameter_names))
                    path = shard_path(directory, name, split, info["image_id"])
                    if read_shard(path, identity, info["image_id"], shape) is not None:
                        print(f"Reused {split} {info['image_id']} {name}", flush=True)
                        continue
                    if array is None:
                        source = roots[split]/info["file_name"]
                        if file_sha256(source) != info["sha256"]:
                            raise ValueError(f"Image changed during estimation: {source}")
                        array = decode_rgb(source)
                        if array.shape[:2] != (info["height"], info["width"]):
                            raise ValueError("Decoded image dimensions differ from manifest.")
                    arrays = estimate_image(specs[name], grid["points"], array, records, cfg["sampling"],
                                            settings, cfg["operators"]["kernel_size"], device)
                    for g in arrays["image_metric"]:
                        metric_diagnostics(g, specs[name].scales, **settings["diagnostics"])
                    write_shard(path, identity, info["image_id"], arrays)
                    print(f"Saved {split} image {index+1}/{len(selection['selected'])}: {info['image_id']} {name}", flush=True)
        from .metric_analysis import analyze_run
        analyze_run(directory)
    print(f"Completed: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
