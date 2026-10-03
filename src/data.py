"""PneumoniaMNIST datasets, transforms and loaders, with start-up checks on the class counts.

Run `python src/data.py` to download the data, check the counts and save two figures that show
what the model sees.
"""
from __future__ import annotations

import numpy as np
import torch
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from config import DATA_DIR, FIG_DIR, Config

# (images, normal = label 0, pneumonia = label 1) per split, recounted from the downloaded file.
EXPECTED_COUNTS = {
    "train": (4708, 1214, 3494),
    "val": (524, 135, 389),
    "test": (624, 234, 390),
}
CLASS_NAMES = ("normal", "pneumonia")


def make_transform(train: bool) -> transforms.Compose:
    """Image -> tensor in [-1, 1]. Training adds small random rotations, shifts and zooms.

    No horizontal flip: a mirrored chest X-ray puts the heart on the wrong side, which never
    happens in a real scan, so it would teach the model something false.
    """
    steps = []
    if train:
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)))
    steps += [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    return transforms.Compose(steps)


def check_counts(labels: np.ndarray, split: str) -> None:
    """Fail loudly if the split does not have exactly the expected number of images per class."""
    labels = labels.reshape(-1)
    got = (len(labels), int((labels == 0).sum()), int((labels == 1).sum()))
    if got != EXPECTED_COUNTS[split]:
        raise AssertionError(f"{split}: expected (n, normal, pneumonia) = {EXPECTED_COUNTS[split]}, got {got}")


def load_split(split: str, train_aug: bool = False, size: int = 28) -> PneumoniaMNIST:
    """Download (once) and return one split. `size=28` is the original MedMNIST resolution."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)   # medmnist raises RuntimeError if root is missing
    ds = PneumoniaMNIST(split=split, transform=make_transform(train_aug), download=True,
                        root=str(DATA_DIR), size=None if size == 28 else size)
    check_counts(ds.labels, split)
    return ds


def pos_weight(train_labels: np.ndarray) -> float:
    """n_normal / n_pneumonia (about 0.347).

    Passed to BCEWithLogitsLoss, it scales down the loss on the majority (pneumonia) class, so the
    two classes contribute equally and a 0.5 probability threshold is a sensible decision point.
    """
    y = train_labels.reshape(-1)
    return float((y == 0).sum() / (y == 1).sum())


def make_train_val_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, float]:
    """Training and validation loaders plus the loss pos_weight. Never touches the test split."""
    train_ds = load_split("train", train_aug=True, size=cfg.size)
    val_ds = load_split("val", train_aug=False, size=cfg.size)
    g = torch.Generator().manual_seed(cfg.seed)   # fixes the shuffle order for this seed
    # num_workers=0: Windows starts worker processes by re-importing the script; the data is tiny,
    # so workers would add failure modes and no speed.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, pos_weight(train_ds.labels)


def make_test_loader(size: int = 28) -> DataLoader:
    """Test loader in dataset order. Only evaluate.py may call this (test-set discipline)."""
    return DataLoader(load_split("test", size=size), batch_size=256, shuffle=False, num_workers=0)


def save_data_figures() -> None:
    """Two figures: 16 training images with labels, and one image under 8 random augmentations."""
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    raw = load_split("train")                      # no augmentation
    aug = load_split("train", train_aug=True)
    to_img = lambda t: (t.squeeze(0) * 0.5 + 0.5).numpy()   # undo Normalize for display

    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for i, ax in enumerate(axes.flat):
        x, y = raw[i]
        ax.imshow(to_img(x), cmap="gray", vmin=0, vmax=1)
        ax.set_title(CLASS_NAMES[int(y[0])], fontsize=9)
        ax.axis("off")
    fig.suptitle("PneumoniaMNIST training images (28x28)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_samples.png", dpi=150)
    plt.close(fig)

    torch.manual_seed(0)
    fig, axes = plt.subplots(1, 9, figsize=(13, 1.9))
    axes[0].imshow(to_img(raw[0][0]), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=9)
    for k, ax in enumerate(axes[1:], start=1):
        ax.imshow(to_img(aug[0][0]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"augmented {k}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_augmentations.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"{'split':<6} {'images':>7} {'normal':>7} {'pneum.':>7} {'share':>6}")
    for split in ("train", "val"):   # test counts are asserted in evaluate.py, the only test reader
        y = load_split(split).labels.reshape(-1)
        print(f"{split:<6} {len(y):>7} {(y == 0).sum():>7} {(y == 1).sum():>7} {y.mean():>6.1%}")
    imgs = load_split("train").imgs / 255.0
    print(f"train pixel mean {imgs.mean():.3f}, std {imgs.std():.3f}")
    print(f"pos_weight = {pos_weight(load_split('train').labels):.3f}")
    save_data_figures()
    print(f"figures saved to {FIG_DIR}")
