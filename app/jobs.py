"""Background jobs: the project's scripts run as separate processes and their output streams into the app.

Training, sweeps, evaluation, tests and benchmarks are exactly the command-line tools from src/,
started with the project's own Python. The app reads their output line by line, so it can show
progress and estimate the time left without changing how the scripts work.
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Callable

from common import ROOT, count_lines

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # no console window pops up on Windows
# One epoch line printed by train.py, e.g. "  ep 12  loss 0.0605  acc 0.957 | val loss 0.0947  auc 0.9938 ..."
EPOCH_RE = re.compile(r"ep\s+(\d+)\s+loss ([\d.]+)\s+acc ([\d.]+) \| val loss ([\d.]+)\s+auc ([\d.]+)\s+"
                      r"acc ([\d.]+) \| grad (\S+) \| ([\d.]+)s")


class Job:
    """One running (or finished) script, e.g. `python src/train.py --mode hybrid --seed 0`."""

    _next_id = 1

    def __init__(self, title: str, python: Path, args: list[str], total_epochs: int | None = None,
                 progress_fn: Callable[[], tuple[int, int]] | None = None) -> None:
        self.id = Job._next_id
        Job._next_id += 1
        self.title, self.python, self.args = title, python, args
        self.total_epochs = total_epochs          # set for a single training run
        self.progress_fn = progress_fn            # set for a sweep: () -> (epochs done, epochs in total)
        self.epochs: list[dict] = []              # parsed epoch lines of a training run
        self.lines: deque[str] = deque(maxlen=20000)
        self.proc: subprocess.Popen | None = None
        self.started = self.ended = None
        self.returncode: int | None = None
        self.stopped = False
        self._q: queue.Queue = queue.Queue()
        self._done0: tuple[float, int] | None = None   # (time, epochs done) when progress was first measured

    def command(self) -> str:
        return " ".join([self.python.name, *self.args])

    def start(self) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
        self.proc = subprocess.Popen([str(self.python), *self.args], cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     bufsize=1, env=env, creationflags=NO_WINDOW)
        self.started = time.time()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self._q.put(line.rstrip("\n").split("\r")[-1])   # keep only the last state of \r progress bars
        self._q.put(None)

    def poll(self) -> list[str]:
        """New output lines since the last call (also records parsed epochs and the exit code)."""
        new = []
        while True:
            try:
                line = self._q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.returncode = self.proc.wait()
                self.ended = time.time()
                break
            new.append(line)
            self.lines.append(line)
            m = EPOCH_RE.search(line)
            if m:
                keys = ("epoch", "train_loss", "train_acc", "val_loss", "val_auc", "val_acc", "grad", "epoch_s")
                self.epochs.append({k: float(v) for k, v in zip(keys, m.groups())})
        return new

    @property
    def running(self) -> bool:
        return self.proc is not None and self.returncode is None

    @property
    def status(self) -> str:
        if self.proc is None:
            return "queued"
        if self.running:
            return "running"
        if self.stopped:
            return "stopped"
        return "finished" if self.returncode == 0 else f"failed (exit {self.returncode})"

    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.ended or time.time()) - self.started

    def progress(self) -> tuple[float | None, float | None]:
        """(fraction done, estimated seconds left); None where it cannot be estimated (yet)."""
        if self.total_epochs:
            done = len(self.epochs)
            if not done:
                return 0.0, None
            per_epoch = sum(e["epoch_s"] for e in self.epochs) / done
            return done / self.total_epochs, per_epoch * (self.total_epochs - done) if self.running else 0.0
        if self.progress_fn:
            done, total = self.progress_fn()
            now = time.time()
            if self._done0 is None:
                self._done0 = (now, done)
            t0, d0 = self._done0
            rate = (done - d0) / (now - t0) if done > d0 and now > t0 else 0.0   # epochs per second, all workers
            eta = (total - done) / rate if rate else None
            return (done / total if total else None), (eta if self.running else 0.0)
        return None, None

    def stop(self) -> None:
        """Stop the job and every process it started (a sweep starts one train.py per run)."""
        if not self.running:
            return
        self.stopped = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=NO_WINDOW)
        else:
            self.proc.terminate()


class JobManager:
    """Keeps the list of jobs and tells listeners about new output every quarter second."""

    def __init__(self, tk_root) -> None:
        self.root = tk_root
        self.jobs: list[Job] = []
        self.listeners: list[Callable[[Job, list[str]], None]] = []
        self._last_status: dict[int, str] = {}
        self.root.after(250, self._pump)

    def submit(self, job: Job) -> Job:
        self.jobs.append(job)
        job.start()
        return job

    def running(self) -> list[Job]:
        return [j for j in self.jobs if j.running]

    def _pump(self) -> None:
        for job in self.jobs:
            new = job.poll() if job.proc else []
            if new or self._last_status.get(job.id) != job.status:
                self._last_status[job.id] = job.status
                for listener in list(self.listeners):
                    try:
                        listener(job, new)
                    except Exception as e:   # a broken view must not stop the other views from updating
                        print(f"listener error: {e!r}")
        self.root.after(250, self._pump)

    def stop_all(self) -> None:
        for j in self.running():
            j.stop()


def stage_progress_fn(stages: list[str]) -> Callable[[], tuple[int, int]]:
    """Progress of a sweep: epochs finished across all its runs, read from their history.csv files."""
    from sweep import STAGES

    runs = {replace(s, seed=k).run_dir(): s.epochs for st in stages for s in STAGES[st][0] for k in STAGES[st][1]}

    def fn() -> tuple[int, int]:
        done = 0
        for d, epochs in runs.items():
            done += epochs if (d / "metrics.json").exists() else max(count_lines(d / "history.csv") - 1, 0)
        return done, sum(runs.values())

    return fn


def format_seconds(s: float | None) -> str:
    if s is None:
        return "estimating..."
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s % 60:02d}s" if m else f"{s % 60}s")
