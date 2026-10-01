"""Generate a tiny synthetic smoke-test set. This is NOT a geometry experiment.

Run from code_setup: python tools/make_demo.py
Then use configs/demo.yaml with the usual prepare/preview commands.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "outputs" / "demo_source"
    for split, indices in (("train", (1,2)), ("valid", (801,))):
        folder = destination / split
        folder.mkdir(parents=True, exist_ok=True)
        for index in indices:
            yy,xx = np.meshgrid(np.linspace(0,1,192),np.linspace(0,1,224),indexing="ij")
            pattern = 0.5+0.3*np.sin(xx*90+index)*np.cos(yy*65)
            array = np.stack((xx,yy,pattern),axis=-1)
            array += np.random.default_rng(index).normal(0,0.02,array.shape)
            Image.fromarray((array.clip(0,1)*255).round().astype(np.uint8)).save(folder/f"{index:04d}.png")
    cfg = yaml.safe_load((root/"configs"/"default.yaml").read_text(encoding="utf-8"))
    cfg["dataset"].update(train_hr=str(destination/"train"), valid_hr=str(destination/"valid"),
                          expected_train_images=2, expected_valid_images=1)
    cfg["sampling"].update(crop_size=128,crops_per_image=2)
    cfg["experiment"]["output_dir"] = "outputs/demo"
    (root/"configs"/"demo.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    print("Created 3 synthetic images and configs/demo.yaml; these cannot support scientific claims.")


if __name__ == "__main__":
    main()
