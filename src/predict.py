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

from config import FINAL_HYBRID, RUNS_DIR, Config, config_from_dict
from data import CLASS_NAMES, make_transform
from models import build_model
from utils import load_json


def load_image(path: Path) -> Image.Image:
    """Greyscale, centre-cropped to a square (as MedMNIST prepared its images), not yet resized."""
    if path.suffix.lower() == ".npy":
        a = np.squeeze(np.load(path))
        if a.ndim != 2:
            raise ValueError(f"expected a 2-D greyscale array, got shape {a.shape}")
        if a.dtype != np.uint8:   # float image: assume values in [0, 1]
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        img = Image.fromarray(a, mode="L")
    else:
        img = Image.open(path).convert("L")
    w, h = img.size
    s = min(w, h)   # a non-square X-ray would otherwise be squashed when resized to a square
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def load_models(run: Path) -> list[tuple[str, Config, torch.nn.Module]]:
    """(seed folder name, config, model in eval mode) for one seed folder or every seed of a model folder."""
    seed_dirs = [run] if (run / "best.pt").exists() else [p.parent for p in sorted(run.glob("seed*/best.pt"))]
    if not seed_dirs:
        raise FileNotFoundError(f"no best.pt under {run}; train first (python src/train.py)")
    models = []
    for d in seed_dirs:
        cfg = config_from_dict(load_json(d / "config.json"))
        model = build_model(cfg)
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))   # weights only (state_dict)
        models.append((d.name, cfg, model.eval()))   # eval: dropout off, batch norm in inference mode
    return models


def to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """Square greyscale image -> (1, 1, size, size) tensor, normalised exactly as in training."""
    return make_transform(train=False)(img.resize((size, size), Image.BILINEAR)).unsqueeze(0)


@torch.no_grad()
def predict_image(img: Image.Image, models: list[tuple[str, Config, torch.nn.Module]]) -> list[float]:
    """P(pneumonia) from each loaded model; their mean is the ensemble prediction."""
    return [torch.sigmoid(m(to_tensor(img, cfg.size))).item() for _, cfg, m in models]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict pneumonia vs normal for one chest X-ray.")
    ap.add_argument("--input", type=Path, required=True, help="PNG/JPG image or .npy array")
    ap.add_argument("--run", type=Path, default=RUNS_DIR / FINAL_HYBRID.run_name(),
                    help="seed folder or model folder (default: all seeds of the final hybrid)")
    args = ap.parse_args()
    models = load_models(args.run)
    probs = predict_image(load_image(args.input), models)
    for (name, _, _), q in zip(models, probs):
        print(f"  {name}: {q:.3f}")
    p = float(np.mean(probs))
    print(f"P(pneumonia) = {p:.3f}  ->  {CLASS_NAMES[int(p > 0.5)]}  (threshold 0.5, mean of {len(probs)} model(s))")
    print("Study project only; not for diagnosis.")
