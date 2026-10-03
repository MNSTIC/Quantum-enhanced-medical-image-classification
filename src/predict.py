"""Classify one chest X-ray:  python src/predict.py --input xray.png

Accepts a PNG/JPG or a .npy array (2-D greyscale; uint8 0-255 or float 0-1). The image is
converted to greyscale, resized to the model's resolution and normalised exactly as in training.
--run may point at one seed folder (runs/hybrid_q8_l3/seed0) or at a model folder
(runs/hybrid_q8_l3), in which case the probabilities of all its seeds are averaged.

Study project only: this must not be used for diagnosis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import FINAL_HYBRID, RUNS_DIR, config_from_dict
from data import CLASS_NAMES, make_transform
from models import build_model
from utils import load_json


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".npy":
        a = np.squeeze(np.load(path))
        if a.ndim != 2:
            raise ValueError(f"expected a 2-D greyscale array, got shape {a.shape}")
        if a.dtype != np.uint8:   # float image: assume values in [0, 1]
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        return Image.fromarray(a, mode="L")
    return Image.open(path).convert("L")


def predict_image(img: Image.Image, run: Path) -> float:
    """P(pneumonia) from one checkpoint, or the mean over all seeds if `run` is a model folder."""
    seed_dirs = [run] if (run / "best.pt").exists() else sorted(run.glob("seed*/best.pt"))
    seed_dirs = [d if d.is_dir() else d.parent for d in seed_dirs]
    if not seed_dirs:
        raise FileNotFoundError(f"no best.pt under {run}; train first (python src/train.py)")
    probs = []
    for d in seed_dirs:
        cfg = config_from_dict(load_json(d / "config.json"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(d / "best.pt"))     # weights only (state_dict)
        model.eval()                                          # dropout off, batch norm in inference mode
        x = make_transform(train=False)(img.resize((cfg.size, cfg.size), Image.BILINEAR)).unsqueeze(0)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(x)).item())
    return float(np.mean(probs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict pneumonia vs normal for one chest X-ray.")
    ap.add_argument("--input", type=Path, required=True, help="PNG/JPG image or .npy array")
    ap.add_argument("--run", type=Path, default=RUNS_DIR / FINAL_HYBRID.run_name(),
                    help="seed folder or model folder (default: all seeds of the final hybrid)")
    args = ap.parse_args()
    p = predict_image(load_image(args.input), args.run)
    print(f"P(pneumonia) = {p:.3f}  ->  {CLASS_NAMES[int(p > 0.5)]}  (threshold 0.5)")
    print("Study project only; not for diagnosis.")
