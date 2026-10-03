"""Time training epochs on the CPU and (if available) the GPU, for the brief's "keep the faster device" check.

    python src/benchmark.py                       # CPU, with the main (CPU-only) environment
    .venv-gpu\\Scripts\\python.exe src/benchmark.py  # CPU and CUDA, with the GPU environment

Appends one row per (environment, device, threads, model) to runs/device_benchmark.csv. Nothing is
saved to runs/<name>/; this only measures speed. The first epoch is discarded (warm-up).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
import torch.nn as nn

from config import FINAL_HYBRID, RUNS_DIR, Config
from data import make_train_val_loaders
from engine import predict, train_one_epoch
from models import build_model
from train import make_optimizer


def time_epochs(cfg: Config, epochs: int) -> list[float]:
    """Seconds per epoch (training pass + validation pass) for `epochs` epochs."""
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(0)
    train_loader, val_loader, pw = make_train_val_loaders(cfg)
    model = build_model(cfg).to(cfg.device)
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=epochs * len(train_loader))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=cfg.device))
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        train_one_epoch(model, train_loader, opt, sched, loss_fn, cfg.grad_clip, cfg.device)
        predict(model, val_loader, loss_fn, cfg.device)
        if cfg.device == "cuda":
            torch.cuda.synchronize()   # GPU work is asynchronous; wait for it before reading the clock
        times.append(time.perf_counter() - t0)
    return times


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 8])
    args = ap.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    models = {"hybrid (final, 8q x 1L)": FINAL_HYBRID, "classical control": Config(mode="classical")}
    rows = []
    for device in devices:
        for threads in args.threads:
            for name, base in models.items():
                cfg = Config(**{**base.__dict__, "device": device, "threads": threads})
                t = time_epochs(cfg, args.epochs)[1:]   # drop the warm-up epoch
                rows.append({"torch": torch.__version__, "device": device, "threads": threads, "model": name,
                             "epoch_s": sum(t) / len(t),
                             "gpu": torch.cuda.get_device_name(0) if device == "cuda" else ""})
                print(f"{torch.__version__:<16} {device:<5} threads {threads:<2} {name:<24} {rows[-1]['epoch_s']:.2f} s/epoch",
                      flush=True)
    out = RUNS_DIR / "device_benchmark.csv"
    pd.concat([pd.read_csv(out), pd.DataFrame(rows)] if out.exists() else [pd.DataFrame(rows)]).to_csv(out, index=False)
