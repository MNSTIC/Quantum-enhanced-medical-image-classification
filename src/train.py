"""Train one model for one seed:  python src/train.py --mode hybrid --seed 0

Writes runs/<name>/seed<k>/: config.json, history.csv, best.pt, val_predictions.csv, metrics.json.
metrics.json is written last, so its presence means "this run finished" (the sweep skips it).

This script never loads the test split. Test scoring lives in evaluate.py only.
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn

from config import Config, config_to_dict, parse_config
from data import make_train_val_loaders
from engine import compute_metrics, predict, train_one_epoch
from models import build_model, count_params, quantum_parameters
from utils import library_versions, save_json, seed_everything


def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.AdamW:
    """AdamW with two parameter groups.

    Circuit angles get a 10x larger learning rate and no weight decay: an angle is periodic, so
    pulling it towards 0 has no regularising meaning, and the angles start small (std 0.1).
    """
    q_ids = {id(p) for p in quantum_parameters(model)}
    classical = [p for p in model.parameters() if id(p) not in q_ids]
    groups = [{"params": classical, "lr": cfg.lr_classical, "weight_decay": cfg.wd_classical}]
    if q_ids:
        groups.append({"params": quantum_parameters(model), "lr": cfg.lr_quantum, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def train(cfg: Config, force: bool = False) -> dict | None:
    out = cfg.run_dir()
    if (out / "metrics.json").exists() and not force:
        print(f"skip {out} (metrics.json exists; use --force to retrain)")
        return None
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    seed_everything(cfg.seed)
    save_json({**config_to_dict(cfg), "versions": library_versions(),
               "started": datetime.now().isoformat(timespec="seconds")}, out / "config.json")

    train_loader, val_loader, pw = make_train_val_loaders(cfg)
    model = build_model(cfg)
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=cfg.epochs * len(train_loader), pct_start=cfg.pct_start)
    # pos_weight < 1 down-weights the majority class (pneumonia) so both classes count equally.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    print(f"{out.relative_to(out.parents[2])}: {count_params(model):,} params "
          f"({count_params(model.probe)} in probed block), pos_weight {pw:.3f}")

    # Checkpoint rule, fixed before any run: best validation AUC, ties broken by lower validation loss.
    best = {"val_auc": -1.0, "val_loss": float("inf")}
    history, best_probs, t_start = [], None, time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, sched, loss_fn, cfg.grad_clip)
        p, y, val_loss = predict(model, val_loader, loss_fn)
        vm = compute_metrics(y, p)
        row = {"epoch": epoch, **{f"lr_{k}": g["lr"] for k, g in zip(("classical", "quantum"), opt.param_groups)}, **tr,
               "val_loss": val_loss, "val_auc": vm["auc"], "val_acc": vm["acc"], "epoch_s": time.time() - t0}
        history.append(row)
        improved = vm["auc"] > best["val_auc"] or (vm["auc"] == best["val_auc"] and val_loss < best["val_loss"])
        if improved:
            best = {"best_epoch": epoch, "val_loss": val_loss, **{f"val_{k}": v for k, v in vm.items()}}
            best_probs = p
            torch.save(model.state_dict(), out / "best.pt")   # state_dict only: TorchLayer models cannot be pickled whole
        print(f"  ep {epoch:2d}  loss {tr['train_loss']:.4f}  acc {tr['train_acc']:.3f} | "
              f"val loss {val_loss:.4f}  auc {vm['auc']:.4f}  acc {vm['acc']:.3f} | "
              f"grad {tr['probe_grad_norm']:.2e} | {row['epoch_s']:.1f}s{'  *' if improved else ''}", flush=True)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame({"label": y, "prob": best_probs}).to_csv(out / "val_predictions.csv", index_label="index")
    metrics = {**best, "n_params": count_params(model), "n_params_probe": count_params(model.probe),
               "train_s": time.time() - t_start, "epoch_s_mean": float(pd.DataFrame(history)["epoch_s"].mean())}
    save_json(metrics, out / "metrics.json")
    print(f"done: best epoch {best['best_epoch']}, val AUC {best['val_auc']:.4f}, "
          f"val ACC {best['val_acc']:.3f}, {metrics['train_s'] / 60:.1f} min")
    return metrics


if __name__ == "__main__":
    cfg, args = parse_config()
    train(cfg, force=args.force)
