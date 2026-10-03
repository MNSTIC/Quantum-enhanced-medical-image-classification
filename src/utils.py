"""Small helpers shared by the scripts: seeding, JSON files, library versions."""
from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch from one number so a run can be repeated exactly.

    The data loader has its own seeded generator (see data.py); the random augmentations and the
    weight initialisation use the global torch generator seeded here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def library_versions() -> dict:
    """Versions saved next to every checkpoint, so a result can be traced to its software."""
    import medmnist
    import pennylane as qml
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pennylane": qml.__version__,
        "medmnist": medmnist.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
