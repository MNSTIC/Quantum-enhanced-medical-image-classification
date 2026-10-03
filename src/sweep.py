"""Run a list of (setting, seed) training jobs, skipping any whose metrics.json already exists.

    python src/sweep.py --stage sweep     # Phase 3: 8 settings x 3 seeds, validation only
    python src/sweep.py --stage final     # Phase 4: chosen hybrid, classical control, no-entanglement; 5 seeds
    python src/sweep.py --stage quanv     # Phase 5.2: quanvolution front end and its classical control
    python src/sweep.py --stage s64       # Phase 5.3: final hybrid and control on 64x64 images

Each job is an independent `train.py` process with a fixed seed and thread count, so running
several at once gives the same numbers as running them one after another. Interrupt at any time
and rerun the same command to resume.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

from config import FINAL_HYBRID, ROOT, Config

DEFAULT = Config(mode="hybrid", n_qubits=8, n_layers=3)

STAGES: dict[str, tuple[list[Config], list[int]]] = {
    # One factor at a time: circuit size grid, then the two on/off ablations at the default size.
    "sweep": ([Config(mode="hybrid", n_qubits=q, n_layers=l) for q in (4, 8) for l in (1, 3, 5)]
              + [replace(DEFAULT, entangle=False), replace(DEFAULT, reupload=False)], [0, 1, 2]),
    "final": ([FINAL_HYBRID, Config(mode="classical", n_qubits=FINAL_HYBRID.n_qubits),
               replace(FINAL_HYBRID, entangle=False)], [0, 1, 2, 3, 4]),
    "quanv": ([Config(mode="quanv"), Config(mode="quanv_classical")], [0, 1, 2, 3, 4]),
    "s64": ([replace(FINAL_HYBRID, size=64), Config(mode="classical", n_qubits=FINAL_HYBRID.n_qubits, size=64)],
            [0, 1, 2, 3, 4]),
}


def to_args(cfg: Config) -> list[str]:
    """Config -> command-line flags for train.py (only the fields that differ from the defaults)."""
    base = Config()
    return [a for f in fields(Config) if getattr(cfg, f.name) != getattr(base, f.name)
            for a in (f"--{f.name}", str(getattr(cfg, f.name)))]


def run_job(cfg: Config) -> str:
    out = cfg.run_dir()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / "train.log", "w") as log:
        rc = subprocess.run([sys.executable, str(ROOT / "src" / "train.py"), *to_args(cfg)],
                            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    status = "ok" if rc == 0 else f"FAILED (exit {rc}, see {out / 'train.log'})"
    return f"{cfg.run_name()}/seed{cfg.seed}: {status}, {(time.time() - t0) / 60:.1f} min"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--workers", type=int, default=8, help="parallel train.py processes (x2 threads each)")
    args = p.parse_args()

    settings, seeds = STAGES[args.stage]
    jobs = [replace(s, seed=k) for s in settings for k in seeds]
    todo = [j for j in jobs if not (j.run_dir() / "metrics.json").exists()]
    print(f"{args.stage}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} "
          f"with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for msg in pool.map(run_job, todo):
            print(msg, flush=True)
