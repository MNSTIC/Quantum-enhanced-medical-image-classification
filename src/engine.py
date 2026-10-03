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
                    sched, loss_fn: nn.Module, clip: float, device: str = "cpu", ema=None) -> dict:
    """One pass over the training data. Returns mean loss, accuracy and the probe gradient norm.

    `ema`, if given, is an EMA object (see EMA below) updated after every optimizer step.
    """
    model.train()   # dropout on, batch norm uses batch statistics
    n, loss_sum, correct, gnorms = 0, 0.0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.squeeze(1).float().to(device)   # (B, 1) int64 -> (B,) float, as the loss expects
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()                      # gradients flow through the simulated circuit too
        gnorms.append(grad_norm(model.probe.parameters()))   # measured before clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()                         # OneCycleLR changes the learning rate every step
        if ema is not None:
            ema.update(model)
        n += len(y)
        loss_sum += loss.item() * len(y)
        correct += int(((logits > 0).float() == y).sum())
    return {"train_loss": loss_sum / n, "train_acc": correct / n, "probe_grad_norm": float(np.mean(gnorms))}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, loss_fn: nn.Module | None = None,
            device: str = "cpu") -> tuple[np.ndarray, np.ndarray, float]:
    """P(pneumonia) for every image, in loader order, plus the true labels and the mean loss."""
    model.eval()    # dropout off, batch norm uses its running statistics
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.squeeze(1).float().to(device)
        logits = model(x)
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * len(y)
        probs.append(torch.sigmoid(logits))
        labels.append(y)
    p, y = torch.cat(probs).cpu().numpy().astype(np.float64), torch.cat(labels).cpu().numpy().astype(int)
    return p, y, loss_sum / len(y)


def shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Move images right by dx and down by dy pixels; the uncovered border is black (-1 after normalising)."""
    h, w = x.shape[-2:]
    padded = nn.functional.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)), value=-1.0)
    return padded[..., max(-dy, 0):max(-dy, 0) + h, max(-dx, 0):max(-dx, 0) + w]


def tta_views(x: torch.Tensor) -> list[torch.Tensor]:
    """Phase 7 test-time augmentation: the batch itself, shifted 2 px in each direction, rotated +-5 degrees.

    All views stay within what training augmentation already showed the model.
    """
    from torchvision.transforms.functional import rotate

    return [x, shift(x, 2, 0), shift(x, -2, 0), shift(x, 0, 2), shift(x, 0, -2),
            rotate(x, 5.0, fill=[-1.0]), rotate(x, -5.0, fill=[-1.0])]


@torch.no_grad()
def predict_tta(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    """P(pneumonia) averaged over the 7 test-time views, in loader order."""
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device)
        probs.append(torch.stack([torch.sigmoid(model(v)) for v in tta_views(x)]).mean(0))
    return torch.cat(probs).cpu().numpy().astype(np.float64)


def weighted_bce(y: np.ndarray, p: np.ndarray, pos_weight: float) -> float:
    """The training loss (BCE with pos_weight), computed from probabilities, so that plain and
    test-time-augmented predictions can be compared on the same footing."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))))


class EMA:
    """Exponential moving average of a model's weights (Phase 7 option).

    After every step: ema = decay * ema + (1 - decay) * weights. The average of the last few
    hundred steps is usually smoother and generalises a little better than the final weights.
    Batch-norm statistics are copied, not averaged, because they are already running averages.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point and "running_" not in k:
                self.state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.state[k].copy_(v)


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
