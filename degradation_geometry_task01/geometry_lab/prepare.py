"""Task 0 CLI: validate sources, freeze crops, save manifests/statistics."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from . import __version__
from .config import load_config, output_path, resolve_path
from .datasets import build_split_manifest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(cfg, overwrite=False):
    output = output_path(cfg)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} is not empty. Use a fresh output_dir or --overwrite.")
    manifests, stats = {}, {}
    for split, key, count_key in (("train", "train_hr", "expected_train_images"),
                                   ("valid", "valid_hr", "expected_valid_images")):
        root = resolve_path(cfg, cfg["dataset"][key])
        print(f"Validating {split}: {root}", flush=True)
        manifests[split], stats[split] = build_split_manifest(
            root, split, cfg["sampling"], cfg["dataset"][count_key])
    train_hashes = {info["pixel_sha256"] for info in manifests["train"]["images"]}
    valid_hashes = {info["pixel_sha256"] for info in manifests["valid"]["images"]}
    if train_hashes & valid_hashes:
        raise ValueError("Train and validation contain identical decoded images; split leakage detected.")
    for split in ("train", "valid"):
        hashes = [info["pixel_sha256"] for info in manifests[split]["images"]]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"{split} contains duplicate decoded images.")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        write_json(output / f"{split}_manifest.json", manifests[split])
        (output / f"{split}_images.txt").write_text(
            "".join(info["file_name"] + "\n" for info in manifests[split]["images"]), encoding="utf-8")
        with (output / f"{split}_content_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "width", "height", "mean_r", "mean_g", "mean_b",
                             "mean_patch_std_r", "mean_patch_std_g", "mean_patch_std_b", "gradient_energy"])
            for rec in stats[split]:
                writer.writerow([rec["image_id"], rec["width"], rec["height"],
                                 *rec["mean_rgb"], *rec["mean_patch_std_rgb"], rec["mean_gradient_energy"]])
    write_json(output / "crop_coordinates.json", {s: manifests[s]["records"] for s in manifests})
    summary = {}
    for split in manifests:
        summary[split] = {
            "images": len(manifests[split]["images"]),
            "crops": len(manifests[split]["records"]),
            "image_weighted_mean_rgb": np.mean([r["mean_rgb"] for r in stats[split]], axis=0).tolist(),
            "image_weighted_mean_gradient_energy": float(np.mean([r["mean_gradient_energy"] for r in stats[split]])),
        }
    write_json(output / "dataset_summary.json", summary)
    resolved = {k: v for k, v in cfg.items() if not k.startswith("_")}
    resolved["dataset"] = dict(cfg["dataset"], train_hr=manifests["train"]["root"],
                               valid_hr=manifests["valid"]["root"])
    resolved["experiment"] = dict(cfg["experiment"], output_dir=str(output))
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    provenance = {"project_version": __version__, "python": platform.python_version(),
                  "platform": platform.platform(), "torch": torch.__version__,
                  "numpy": np.__version__, "Pillow": importlib.metadata.version("Pillow"),
                  "manifest_sha256": {split: hashlib.sha256((output / f"{split}_manifest.json").read_bytes()).hexdigest()
                                      for split in manifests}}
    write_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2))
    print(f"Prepared Task 0 artifacts: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-hr", help="Override dataset.train_hr")
    parser.add_argument("--valid-hr", help="Override dataset.valid_hr")
    parser.add_argument("--overwrite", action="store_true", help="Replace Task 0 artifacts in the configured output directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.train_hr:
        cfg["dataset"]["train_hr"] = args.train_hr
    if args.valid_hr:
        cfg["dataset"]["valid_hr"] = args.valid_hr
    torch.set_num_threads(cfg["experiment"]["torch_threads"])
    prepare(cfg, args.overwrite)


if __name__ == "__main__":
    main()
