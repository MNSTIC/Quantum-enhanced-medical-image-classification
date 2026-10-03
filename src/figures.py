"""Plot functions used by evaluate.py. Each takes already-computed numbers and writes one PNG.

Colours: one fixed colour per model, everywhere (blue = hybrid, orange = classical control,
aqua = third model), from a palette checked for colour-blind separation. Text stays dark grey.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # write files only; no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(histories: dict[str, list[pd.DataFrame]], path: Path) -> None:
    """Mean +- std over seeds of training loss, validation loss and validation AUC per epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("train_loss", "Training loss"), ("val_loss", "Validation loss"), ("val_auc", "Validation AUC")]
    for (col, title), ax in zip(panels, axes):
        for color, (label, hs) in zip(SERIES, histories.items()):
            v = np.stack([h[col].to_numpy() for h in hs])
            ep, m, s = hs[0]["epoch"].to_numpy(), v.mean(0), v.std(0, ddof=1) if len(hs) > 1 else 0 * v[0]
            ax.plot(ep, m, color=color, label=f"{label} (n={len(hs)})")
            ax.fill_between(ep, m - s, m + s, color=color, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[2].set_ylim(0.95, 1.0)
    axes[0].legend(fontsize=9)
    fig.suptitle("Training curves: mean over seeds, band = 1 standard deviation", fontsize=11)
    _save(fig, path)


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> None:
    """ROC of each seed-ensemble on the test set. `curves[label] = (y, p, auc)`."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for color, (label, (y, p, auc)) in zip(SERIES, curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=f"{label}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_title("Test ROC, 5-seed ensembles")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, path)


def confusion_matrices(cms: dict[str, dict], path: Path) -> None:
    """Confusion matrix (threshold 0.5) of each seed-ensemble on the test set."""
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4))
    for ax, (label, m) in zip(np.atleast_1d(axes), cms.items()):
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(mat / mat.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = mat[i, j] / mat[i].sum()
                ax.text(j, i, f"{mat[i, j]}\n({frac:.0%} of row)", ha="center", va="center",
                        color="white" if frac > 0.6 else INK, fontsize=10)
        ax.set_xticks([0, 1], ["normal", "pneumonia"])
        ax.set_yticks([0, 1], ["normal", "pneumonia"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(label, fontsize=10)
        ax.grid(False)
    fig.suptitle("Test confusion matrices, 5-seed ensembles, threshold 0.5", fontsize=11)
    _save(fig, path)


def score_bars(scores: dict[str, dict], baselines: dict[str, tuple[float, float]], path: Path) -> None:
    """Test AUC and ACC per model (mean, error bar = seed std, dots = single seeds) + published lines.

    `scores[label] = {"auc": array over seeds, "acc": array over seeds}`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = list(scores)
    for k, (metric, ax) in enumerate(zip(("auc", "acc"), axes)):
        for i, (label, color) in enumerate(zip(labels, SERIES)):
            v = np.asarray(scores[label][metric])
            ax.bar(i, v.mean(), width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color=INK, capsize=5, linewidth=1.2)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=14, color=INK, zorder=3)
            ax.text(i, v.mean() + 0.006, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        for name, vals in baselines.items():
            ax.axhline(vals[k], color=INK2, linestyle="--", linewidth=1)
            ax.text(len(labels) - 0.45, vals[k], name, fontsize=8, color=INK2, va="center")
        ax.set_xticks(range(len(labels)), [lab.replace(", ", ",\n") for lab in labels], fontsize=9)
        ax.set_xlim(-0.5, len(labels) + 1.4)
        ax.set_ylim(0.80, 1.0)
        ax.set_title(f"Test {metric.upper()} (threshold 0.5 for ACC)")
    fig.suptitle("Test scores per model: bar = mean of seeds, whisker = 1 std, dots = seeds; "
                 "dashed = published baselines", fontsize=10)
    _save(fig, path)


def sweep_bars(table: pd.DataFrame, path: Path) -> None:
    """Validation AUC (mean +- std over seeds) for every Phase 3 setting."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(table))
    ax.bar(x, table["val_auc_mean"], width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, table["val_auc_mean"], yerr=table["val_auc_std"], fmt="none", color=INK, capsize=4, linewidth=1.2)
    ax.set_xticks(x, table["setting"], rotation=30, ha="right", fontsize=9)
    lo = (table["val_auc_mean"] - table["val_auc_std"]).min()
    ax.set_ylim(max(0.9, lo - 0.005), 1.0)
    ax.set_ylabel("validation AUC")
    ax.set_title("Phase 3 sweep: validation AUC, mean of 3 seeds, whisker = 1 std")
    _save(fig, path)


def noise_curves(df: pd.DataFrame, path: Path) -> None:
    """Test AUC and ACC of the trained hybrid vs depolarising probability p (mean +- std over seeds).

    `df` rows: setting ('exact', 'shots=1000', 'p=0.001', ...), p, metric means and stds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    noisy = df[df["p"] > 0].sort_values("p")
    for metric, ax in zip(("auc", "acc"), axes):
        ax.errorbar(noisy["p"], noisy[f"{metric}_mean"], yerr=noisy[f"{metric}_std"], color=SERIES[0],
                    marker="o", markersize=7, capsize=4, label="depolarising noise after each layer")
        for (_, r), style in zip(df[df["p"] == 0].iterrows(), ("-", ":")):
            ax.axhline(r[f"{metric}_mean"], color=INK2, linestyle=style, linewidth=1.2, label=r["setting"])
        ax.set_xscale("log")
        ax.set_xlabel("depolarising probability p per qubit per layer")
        ax.set_title(f"Test {metric.upper()}")
    axes[0].legend(fontsize=8)
    fig.suptitle("Trained hybrid evaluated under finite shots and noise (no retraining), mean of seeds", fontsize=10)
    _save(fig, path)
