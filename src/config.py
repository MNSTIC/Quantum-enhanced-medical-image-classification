"""All hyperparameters in one dataclass, plus a command-line parser that can override any of them.

Why a dataclass: every run saves its config to `config.json`, and `evaluate.py` / `predict.py`
rebuild the exact same model from that file. One source of truth avoids "which settings was this
checkpoint trained with?" questions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (the folder that holds src/)
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "figures"

MODES = ("hybrid", "classical", "quanv", "quanv_classical")


@dataclass
class Config:
    # --- model ---
    mode: str = "hybrid"           # hybrid | classical | quanv | quanv_classical (see models.py)
    n_qubits: int = 8              # width of the 8->8 middle block (qubits for hybrid, units for classical)
    n_layers: int = 3              # number of trainable circuit layers
    entangle: bool = True          # False = ablation: no CNOTs, qubits never interact
    reupload: bool = True          # False = ablation: encode the data only before the first layer
    quanv_layers: int = 2          # circuit layers in the 4-qubit quanvolution filter (Phase 5.2 only)
    dropout: float = 0.3
    # --- data ---
    size: int = 28                 # 28 = original MedMNIST; 64 = MedMNIST+ (Phase 5.3 only)
    # --- optimisation (section 8 of the brief) ---
    epochs: int = 30
    batch_size: int = 64
    lr_classical: float = 1e-3
    lr_quantum: float = 1e-2
    wd_classical: float = 1e-4
    pct_start: float = 0.15
    grad_clip: float = 1.0
    ema: float = 0.0               # Phase 7: decay of an exponential moving average of the weights; 0 = off
    # --- bookkeeping ---
    seed: int = 0
    threads: int = 2               # torch CPU threads; 1 and 2 give bit-identical results, 4 does not
    device: str = "cpu"            # "cpu" or "cuda"; every reported result uses the CPU (see DECISIONS.md)
    name: str = ""                 # run-group name; empty = derived from the settings (see run_name)

    def run_name(self) -> str:
        """Folder name for this setting, e.g. 'hybrid_q8_l3' or 'hybrid_q8_l3_noent'.

        Runs that differ only in seed share a name, so runs/<name>/seed<k>/ groups the seeds.
        """
        if self.name:
            return self.name
        if self.mode == "hybrid":
            n = f"hybrid_q{self.n_qubits}_l{self.n_layers}"
            n += "" if self.entangle else "_noent"
            n += "" if self.reupload else "_noreup"
        elif self.mode == "classical":
            n = f"classical_q{self.n_qubits}"
        elif self.mode == "quanv":
            n = f"quanv_l{self.quanv_layers}"
        else:
            n = "quanv_classical"
        if self.size != 28:
            n += f"_s{self.size}"
        if self.epochs != 30:
            n += f"_e{self.epochs}"
        if self.ema:
            n += f"_ema{self.ema:g}"
        if self.device != "cpu":   # GPU arithmetic differs slightly, so GPU runs never share a CPU run's folder
            n += f"_{self.device}"
        return n

    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name() / f"seed{self.seed}"


# The hybrid configuration chosen in Phase 3 on validation data only (reason in DECISIONS.md):
# all circuits tied on validation AUC, so the smallest circuit that also tied on validation loss.
# Phase 4 trains it with 5 seeds; evaluate.py scores it on the test set.
FINAL_HYBRID = Config(mode="hybrid", n_qubits=8, n_layers=1)
# Declared before any test scoring: the default 8x3 circuit, whose 72 angles match the classical
# control's 72 weights, is also trained with 5 seeds as a secondary, parameter-matched comparison.
MATCHED_HYBRID = Config(mode="hybrid", n_qubits=8, n_layers=3)


def _str2bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "on"):
        return True
    if s.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def parse_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    """Build a Config from command-line flags. Every dataclass field becomes a --flag."""
    p = argparse.ArgumentParser(description="Train one model for one seed.")
    for f in fields(Config):
        kind = {"bool": _str2bool, "int": int, "float": float}.get(f.type, str)
        p.add_argument(f"--{f.name}", type=kind, default=f.default)
    p.add_argument("--force", action="store_true", help="retrain even if metrics.json already exists")
    args = p.parse_args(argv)
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    if cfg.mode not in MODES:
        p.error(f"--mode must be one of {MODES}")
    return cfg, args


def config_from_dict(d: dict) -> Config:
    """Rebuild a Config from a saved config.json (ignores extra keys such as library versions)."""
    names = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in names})


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)
