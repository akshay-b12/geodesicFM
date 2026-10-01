"""Task 1 CLI: operator progression grids, raw tensors and complete metadata."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .config import load_config, output_path, resolve_path
from .datasets import ManifestDataset, core_crop
from .degradations import (affine_intensity, gaussian_blur, anisotropic_blur,
                           gaussian_noise, make_epsilon, haze, blur_noise, blur_haze)
from .prepare import write_json


def display_array(x, color_space):
    # Never feed this clipped/quantized display tensor back into the study.
    x = x.detach().cpu().clamp(0, 1)
    if color_space == "linear_rgb":
        x = torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1 / 2.4) - 0.055)
    return (x.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def save_grid(path, clean, outputs, labels, color_space):
    height, width = clean.shape[-2:]
    arrays = [display_array(clean, color_space)] + [display_array(y, color_space) for y in outputs]
    labels = ["clean"] + labels
    columns = min(4, len(arrays))
    rows = math.ceil(len(arrays) / columns)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = []
    for label in labels:
        lines, line = [], ""
        for clause in label.split(", "):
            candidate = f"{line}, {clause}" if line else clause
            if line and probe.textbbox((0, 0), candidate)[2] > width - 8:
                lines.append(line)
                line = clause
            else:
                line = candidate
        lines.append(line)
        wrapped.append(lines)
    caption_height = 12 * max(len(lines) for lines in wrapped) + 8
    canvas = Image.new("RGB", (columns * width, rows * (height + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (array, lines) in enumerate(zip(arrays, wrapped)):
        col, row = idx % columns, idx // columns
        x0, y0 = col * width, row * (height + caption_height)
        canvas.paste(Image.fromarray(array), (x0, y0))
        for line_index, line in enumerate(lines):
            draw.text((x0 + 4, y0 + height + 4 + 12 * line_index), line, fill="black")
    canvas.save(path)


def preview(cfg, overwrite=False):
    output = output_path(cfg)
    manifest_path = output / "train_manifest.json"
    dtype = getattr(torch, cfg["experiment"]["dtype"])
    override = cfg["dataset"]["train_hr"]
    ds = ManifestDataset(manifest_path, root_override=resolve_path(cfg, override) if override else None, dtype=dtype)
    if ds.manifest["sampling"] != cfg["sampling"]:
        raise ValueError("Sampling/color-space config changed; prepare a fresh manifest first.")
    image_idx, crop_idx = cfg["preview"]["image_index"], cfg["preview"]["crop_index"]
    matches = [idx for idx, rec in enumerate(ds.records)
               if rec["image_index"] == image_idx and rec["crop_index"] == crop_idx]
    if len(matches) != 1:
        raise ValueError("Requested image_index/crop_index is not in the manifest.")
    item = ds[matches[0]]
    x = item["image"].to(cfg["experiment"]["device"])
    h, size = item["halo"], item["crop_size"]
    clean = core_crop(x, h, size)
    k = cfg["operators"]["kernel_size"]
    lo, hi = cfg["operators"]["sigma_domain"]
    sigmas = np.linspace(lo, hi, 5).tolist()
    mid = (lo + hi) / 2
    epsilon = make_epsilon(x, cfg["preview"]["noise_seed"])
    unit_depth = torch.ones((1, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)
    toy_depth = torch.linspace(0.25, 1.75, x.shape[-1], dtype=x.dtype, device=x.device)
    toy_depth = toy_depth[None, None, :].expand(1, x.shape[-2], -1).contiguous()
    definitions = {
        "isotropic_blur": (lambda xi: gaussian_blur(x, xi, k), [[s] for s in sigmas], ["sigma"]),
        "anisotropic_axes": (lambda xi: anisotropic_blur(x, xi, k), [[s, lo, 0.0] for s in sigmas],
                             ["sigma_x", "sigma_y", "theta_rad"]),
        "anisotropic_rotation": (lambda xi: anisotropic_blur(x, xi, k), [[hi, lo, a] for a in (0, math.pi/6, math.pi/3, math.pi/2)],
                                 ["sigma_x", "sigma_y", "theta_rad"]),
        "affine_intensity": (lambda xi: affine_intensity(x, xi), [[a, b] for a, b in ((0.7,0), (1,0), (1.3,0), (1,-0.1), (1,0.1))], ["a", "b"]),
        "gaussian_noise": (lambda xi: gaussian_noise(x, xi, epsilon=epsilon), [[s] for s in (0,0.02,0.05,0.1)], ["sigma_noise"]),
        "haze_unit_depth": (lambda xi: haze(x, xi, depth=unit_depth), [[b,0.9] for b in (0,0.25,0.5,1,2)], ["beta", "A"]),
        "haze_toy_depth": (lambda xi: haze(x, xi, depth=toy_depth), [[b,0.9] for b in (0,0.25,0.5,1,2)], ["beta", "A"]),
        "blur_then_noise": (lambda xi: blur_noise(x, xi, epsilon=epsilon, kernel_size=k), [[s,0.05] for s in sigmas], ["sigma_blur", "sigma_noise"]),
        "noise_then_blur": (lambda xi: blur_noise(x, xi, epsilon=epsilon, kernel_size=k, order="noise_then_blur"), [[s,0.05] for s in sigmas], ["sigma_blur", "sigma_noise"]),
        "blur_then_haze_toy_depth": (lambda xi: blur_haze(x, xi, depth=toy_depth, kernel_size=k), [[mid,b,0.9] for b in (0,0.25,0.5,1,2)], ["sigma_blur", "beta", "A"]),
        "haze_then_blur_toy_depth": (lambda xi: blur_haze(x, xi, depth=toy_depth, kernel_size=k, order="haze_then_blur"), [[mid,b,0.9] for b in (0,0.25,0.5,1,2)], ["sigma_blur", "beta", "A"]),
    }
    destination = output / "previews"
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError("Previews already exist. Choose a fresh output_dir or --overwrite.")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"source_image_id": item["image_id"], "crop_index": item["crop_index"],
                "record": ds.records[matches[0]], "color_space": cfg["sampling"]["color_space"],
                "dtype": cfg["experiment"]["dtype"], "device": str(x.device), "kernel_size": k,
                "noise_seed": cfg["preview"]["noise_seed"], "noise_torch_version": torch.__version__,
                "train_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "note": "PNG previews clip to [0,1]. Saved raw tensors retain all values. Toy depth is not ground-truth scene depth.",
                "operators": {}}
    tensors = {"clean_context": x.cpu(), "clean_core": clean.cpu(),
               "epsilon_context": epsilon.cpu(), "unit_depth_context": unit_depth.cpu(),
               "toy_depth_context": toy_depth.cpu()}
    with torch.no_grad():
        for name, (fn, values, parameter_names) in definitions.items():
            ys = [core_crop(fn(torch.as_tensor(p, dtype=x.dtype, device=x.device)), h, size) for p in values]
            labels = [", ".join(f"{key}={value:.2f}" for key, value in zip(parameter_names, p)) for p in values]
            save_grid(destination / f"{name}.png", clean, ys, labels, cfg["sampling"]["color_space"])
            tensors[name] = torch.stack(ys).cpu()
            metadata["operators"][name] = {"parameter_names": parameter_names, "parameter_vectors": values,
                                           "raw_min": min(float(y.min()) for y in ys),
                                           "raw_max": max(float(y.max()) for y in ys)}
    torch.save(tensors, destination / "raw_tensors.pt")
    write_json(destination / "preview_metadata.json", metadata)
    print(f"Saved {len(definitions)} grids, raw tensors and metadata: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg["experiment"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    preview(cfg, args.overwrite)


if __name__ == "__main__":
    main()
