"""Image-level manifests with deterministic crops and physical blur context.

Each sampling key has its own SHA256-derived seed. Adding or reordering images
does not alter other images' crop coordinates. Bootstrap the image_id groups,
never the crop records independently.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def stable_seed(*parts):
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb(path):
    # DIV2K is 8-bit RGB PNG. Refuse silent grayscale/alpha/16-bit conversion.
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != "RGB":
            raise ValueError(f"Expected 8-bit RGB PNG: {path}; got {im.format}/{im.mode}")
        array = np.asarray(im).copy()
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Unsupported image representation: {path}")
    return array


def rgb_tensor(array, color_space="srgb", dtype=torch.float64):
    x = torch.from_numpy(array.copy()).permute(2, 0, 1).to(dtype) / 255.0
    if color_space == "linear_rgb":
        x = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))
    elif color_space != "srgb":
        raise ValueError("Unknown color space.")
    return x


def core_crop(x, halo, crop_size):
    """Apply operators to context first, then measure only this interior core."""
    if x.shape[-2] < halo + crop_size or x.shape[-1] < halo + crop_size:
        raise ValueError("Tensor is too small for requested core.")
    return x[..., halo:halo + crop_size, halo:halo + crop_size]


def build_split_manifest(root, split, sampling, expected_count=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise ValueError(f"No PNG images found directly inside {root}")
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(f"{split}: expected {expected_count} images, found {len(paths)}. "
                         "For a deliberate pilot set, change expected_*_images explicitly.")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Image stems must be unique within a split.")
    size, halo = sampling["crop_size"], sampling["halo"]
    records, images, stats = [], [], []
    for image_index, path in enumerate(paths):
        array = decode_rgb(path)
        height, width = array.shape[:2]
        minimum = size + 2 * halo
        if min(height, width) < minimum:
            raise ValueError(f"{path.name} is {width}x{height}; need >= {minimum} on each axis. "
                             "No silent resizing or padded clean crops is permitted.")
        image_id = f"{split}/{path.stem}"
        digest = file_sha256(path)
        # Catch equal decoded pixels even if PNG encodings differ across splits.
        pixel_digest = hashlib.sha256(
            f"{height},{width},3:".encode() + array.tobytes()).hexdigest()
        images.append({"image_id": image_id, "file_name": path.name,
                       "sha256": digest, "pixel_sha256": pixel_digest,
                       "width": width, "height": height})
        core_means, core_stds, core_gradients = [], [], []
        for crop_index in range(sampling["crops_per_image"]):
            seed = stable_seed(sampling["seed"], image_id, crop_index)
            rng = random.Random(seed)
            top = rng.randrange(halo, height - size - halo + 1)
            left = rng.randrange(halo, width - size - halo + 1)
            records.append({"image_id": image_id, "image_index": image_index,
                            "file_name": path.name, "crop_index": crop_index,
                            "top": top, "left": left, "crop_size": size,
                            "halo": halo, "sampling_seed": seed})
            clean = rgb_tensor(array[top:top + size, left:left + size], sampling["color_space"])
            core_means.append(clean.mean((1, 2)).numpy())
            core_stds.append(clean.std((1, 2), correction=0).numpy())
            gradient = ((clean[:, 1:] - clean[:, :-1]).square().mean()
                        + (clean[:, :, 1:] - clean[:, :, :-1]).square().mean()) / 2
            core_gradients.append(float(gradient))
        stats.append({"image_id": image_id, "width": width, "height": height,
                      "mean_rgb": np.mean(core_means, axis=0).tolist(),
                      "mean_patch_std_rgb": np.mean(core_stds, axis=0).tolist(),
                      "mean_gradient_energy": float(np.mean(core_gradients))})
    manifest = {"schema_version": 1, "split": split, "root": str(root),
                "sampling": dict(sampling), "images": images, "records": records}
    return manifest, stats


class ManifestDataset(Dataset):
    """Returns a context tensor and metadata; no random transforms or resizing.

    root_override makes manifests portable between Windows/Linux. Verification
    hashes every original file once when this dataset is constructed.
    """
    def __init__(self, manifest_path, root_override=None, dtype=torch.float64,
                 verify_files=True):
        with Path(manifest_path).open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported manifest version.")
        self.root = Path(root_override or self.manifest["root"])
        self.records = self.manifest["records"]
        self.dtype = dtype
        if verify_files:
            for info in self.manifest["images"]:
                path = self.root / info["file_name"]
                if file_sha256(path) != info["sha256"]:
                    raise ValueError(f"Image changed since manifest preparation: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        array = decode_rgb(self.root / rec["file_name"])
        t, l, s, h = (rec[k] for k in ("top", "left", "crop_size", "halo"))
        patch = array[t - h:t + s + h, l - h:l + s + h]
        if patch.shape[:2] != (s + 2*h, s + 2*h):
            raise ValueError("Manifest crop lies outside the source image.")
        image = rgb_tensor(patch, self.manifest["sampling"]["color_space"], self.dtype)
        return {"image": image, "image_id": rec["image_id"],
                "crop_index": rec["crop_index"], "halo": h, "crop_size": s}
