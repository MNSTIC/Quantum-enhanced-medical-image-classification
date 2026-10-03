"""System tab: what the project runs on, the test suite, the CPU-vs-GPU benchmark and disk use."""
from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from tkinter import ttk

from common import (DATA_DIR, FIG_DIR, INK2, ROOT, RUNS_DIR, Table, append_log, gpu_available, log_box,
                    python_for, read_csv_rows, section)
from jobs import NO_WINDOW, Job, JobManager, format_seconds


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2 ** 20 if path.exists() else 0.0


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout.strip()
        return out or "none found"
    except (OSError, subprocess.TimeoutExpired):
        return "none found (nvidia-smi not available)"


def version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


class SystemTab(ttk.Frame):
    def __init__(self, master, jobs: JobManager) -> None:
        super().__init__(master, padding=8)
        self.jobs, self.job = jobs, None
        jobs.listeners.append(self.on_job)

        top = ttk.Frame(self)
        top.pack(fill="x")
        env = section(top, "Environment")
        env.pack(side="left", fill="both", expand=True)
        gpu_torch = next((p.name.removeprefix("torch-").removesuffix(".dist-info")
                          for p in (ROOT / ".venv-gpu" / "Lib" / "site-packages").glob("torch-*.dist-info")), "")
        rows = [
            ["Operating system", platform.platform()],
            ["CPU", f"{platform.processor() or 'unknown'} ({os.cpu_count()} logical cores)"],
            ["GPU", gpu_name()],
            ["Python", platform.python_version()],
            ["PyTorch (main environment)", version("torch")],
            ["PyTorch (GPU environment)", gpu_torch or "not set up (.venv-gpu missing)"],
            ["PennyLane", version("pennylane")],
            ["MedMNIST", version("medmnist")],
            ["NumPy / scikit-learn", f"{version('numpy')} / {version('scikit-learn')}"],
        ]
        t = Table(env, ["item", "value"], {"item": 190, "value": 520}, height=len(rows))
        t.pack(fill="both", expand=True)
        t.set_rows(rows)

        disk = section(top, "Disk use")
        disk.pack(side="left", fill="y", padx=(8, 0))
        for name, path in (("data/", DATA_DIR), ("runs/", RUNS_DIR), ("figures/", FIG_DIR)):
            ttk.Label(disk, text=f"{name:<10} {folder_mb(path):8.1f} MB", font=("Consolas", 10)).pack(anchor="w")

        actions = section(self, "Checks")
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Run the test suite (pytest)", command=self.run_tests).pack(side="left")
        ttk.Button(actions, text="Benchmark CPU vs GPU", command=self.run_benchmark).pack(side="left", padx=6)
        self.status = ttk.Label(actions, text="", foreground=INK2)
        self.status.pack(side="left", padx=8)
        ttk.Label(self, foreground=INK2, wraplength=1100, justify="left",
                  text="Reported results all use the CPU, 2 threads per run (bit-for-bit reproducible). The GPU "
                       "environment is optional and only speeds up single runs; GPU arithmetic differs slightly, "
                       "so GPU runs are saved in separate '_cuda' folders.").pack(anchor="w")

        bench = section(self, "Seconds per epoch (runs/device_benchmark.csv)")
        bench.pack(fill="x")
        self.bench = Table(bench, ["torch", "device", "threads", "model", "s/epoch", "gpu"], {"model": 200, "gpu": 260}, height=6)
        self.bench.pack(fill="x")
        self.log = log_box(self, height=12)
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.load_bench()

    def load_bench(self) -> None:
        self.bench.set_rows([[r["torch"], r["device"], r["threads"], r["model"], f"{float(r['epoch_s']):.2f}", r.get("gpu", "")]
                             for r in read_csv_rows(RUNS_DIR / "device_benchmark.csv")])

    def run_tests(self) -> None:
        self.job = self.jobs.submit(Job("tests (pytest)", python_for("cpu"), ["-m", "pytest", "-q", "-p", "no:cacheprovider"]))

    def run_benchmark(self) -> None:
        dev = "cuda" if gpu_available() else "cpu"
        self.job = self.jobs.submit(Job("benchmark", python_for(dev), ["src/benchmark.py"]))

    def on_job(self, job: Job, new: list[str]) -> None:
        if job is not self.job:
            return
        append_log(self.log, new)
        self.status.configure(text=f"{job.title}: {job.status}, {format_seconds(job.elapsed())}")
        if not job.running:
            self.load_bench()
