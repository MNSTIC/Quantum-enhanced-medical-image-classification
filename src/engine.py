"""Training step, prediction and metrics. Shared by train.py and evaluate.py."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


def grad_norm(params) -> float:
    """L2 norm of the gradients of `params`. A value collapsing towards 0 signals a barren plateau."""
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.cat(grads).norm()) if grads else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                    sched, loss_fn: nn.Module, clip: float) -> dict:
    """One pass over the training data. Returns mean loss, accuracy and the probe gradient norm."""
    model.train()   # dropout on, batch norm uses batch statistics
    n, loss_sum, correct, gnorms = 0, 0.0, 0, []
    for x, y in loader:
        y = y.squeeze(1).float()             # (B, 1) int64 -> (B,) float, as the loss expects
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()                      # gradients flow through the simulated circuit too
        gnorms.append(grad_norm(model.probe.parameters()))   # measured before clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()                         # OneCycleLR changes the learning rate every step
        n += len(y)
        loss_sum += loss.item() * len(y)
        correct += int(((logits > 0).float() == y).sum())
    return {"train_loss": loss_sum / n, "train_acc": correct / n, "probe_grad_norm": float(np.mean(gnorms))}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, loss_fn: nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """P(pneumonia) for every image, in loader order, plus the true labels and the mean loss."""
    model.eval()    # dropout off, batch norm uses its running statistics
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        y = y.squeeze(1).float()
        logits = model(x)
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * len(y)
        probs.append(torch.sigmoid(logits))
        labels.append(y)
    p, y = torch.cat(probs).numpy().astype(np.float64), torch.cat(labels).numpy().astype(int)
    return p, y, loss_sum / len(y)


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """AUC plus threshold metrics. Pneumonia (label 1) is the positive class.

    `p > threshold` (strictly greater) matches the official MedMNIST Evaluator.
    """
    pred = (p > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)                         # share of pneumonia cases caught (recall)
    spec = tn / (tn + fp)                         # share of normal cases correctly cleared
    prec = tp / (tp + fp) if tp + fp else 0.0     # share of pneumonia alarms that were right
    return {
        "auc": float(roc_auc_score(y, p)),
        "acc": float((pred == y).mean()),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(2 * prec * sens / (prec + sens)) if prec + sens else 0.0,
        "balanced_acc": float((sens + spec) / 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """95% percentile bootstrap intervals for AUC and ACC: resample the test images with replacement.

    This measures how much the score would move with a different draw of 624 test patients; it
    says nothing about training randomness (that is what the 5 seeds are for).
    """
    rng = np.random.default_rng(seed)
    aucs, accs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[i], p[i]))
        accs.append(((p[i] > 0.5) == y[i]).mean())
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"auc_ci": [float(v) for v in np.percentile(aucs, q)],
            "acc_ci": [float(v) for v in np.percentile(accs, q)]}
