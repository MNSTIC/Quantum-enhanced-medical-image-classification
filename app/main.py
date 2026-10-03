"""Quantum Pneumonia Lab: a desktop control centre for this project.

    .venv\\Scripts\\pythonw.exe app\\main.py      (or double-click QuantumPneumoniaLab.exe)

Every tab drives the project's own code: training runs are src/train.py, stages are src/sweep.py,
results come from src/evaluate.py, predictions from src/predict.py. The app adds a window, live
progress and time estimates on top; it never computes a result differently from the scripts.
"""
from __future__ import annotations

import ctypes
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (FONT_BOLD, FONT_TITLE, INK2, ROOT, RUNS_DIR, Table, append_log, apply_theme, log_box,  # noqa: E402
                    markdown_to_text, read_csv_rows, results_sections, scan_runs, section, set_text)
from jobs import JobManager, format_seconds  # noqa: E402
from tab_data import DataTab, npz_path  # noqa: E402
from tab_learn import LearnTab  # noqa: E402
from tab_predict import PredictTab  # noqa: E402
from tab_results import ResultsTab  # noqa: E402
from tab_runs import RunsTab  # noqa: E402
from tab_system import SystemTab  # noqa: E402
from tab_train import TrainTab  # noqa: E402

GROUP_LABELS = {"hybrid_q8_l1": "Hybrid, 8 qubits x 1 layer (final)", "classical_q8": "Classical control",
                "hybrid_q8_l1_noent": "Hybrid, no entanglement", "hybrid_q8_l3": "Hybrid, 8 x 3 (parameter-matched)"}


class Overview(ttk.Frame):
    """Home tab: where the project stands and the headline answer, in one screen."""

    def __init__(self, master, app: "App") -> None:
        super().__init__(master, padding=12)
        self.app = app
        ttk.Label(self, text="Quantum-enhanced pneumonia classification (PneumoniaMNIST, PennyLane)",
                  font=FONT_TITLE).pack(anchor="w")
        ttk.Label(self, foreground=INK2, text="Hybrid quantum-classical classifier vs a classical control of the same "
                  "size, simulated on this laptop. Study project only, not for diagnosis.").pack(anchor="w", pady=(0, 8))
        cards = ttk.Frame(self)
        cards.pack(fill="x")
        self.cards = {}
        for k, title in enumerate(("Data", "Training runs", "Test results", "Jobs running")):
            f = section(cards, title)
            f.grid(row=0, column=k, sticky="nsew", padx=4)
            cards.columnconfigure(k, weight=1)
            self.cards[title] = ttk.Label(f, text="", font=FONT_BOLD, justify="left")
            self.cards[title].pack(anchor="w")

        head = section(self, "Headline (test set, 624 images, mean ± std over seeds; from runs/summary.csv)")
        head.pack(fill="x", pady=8)
        self.headline = Table(head, ["model", "seeds", "test AUC", "test ACC", "sensitivity", "specificity",
                                     "ensemble AUC", "ensemble ACC"], {"model": 260}, height=4)
        self.headline.pack(fill="x")
        self.verdict = tk.Text(head, height=9, font=("Consolas", 9), wrap="none", state="disabled")
        self.verdict.pack(fill="x", pady=(6, 0))

        quick = section(self, "Quick actions")
        quick.pack(fill="x")
        for text, cmd in (("Train a model", lambda: app.show("Train")), ("Rebuild results", lambda: app.tabs["Results"].run([])),
                          ("Predict an X-ray", lambda: app.show("Predict")), ("How it works", lambda: app.show("Learn")),
                          ("Open REPORT.md", lambda: os.startfile(ROOT / "REPORT.md")),
                          ("Open project folder", lambda: os.startfile(ROOT))):
            ttk.Button(quick, text=text, command=cmd).pack(side="left", padx=3)
        self.refresh()

    def refresh(self) -> None:
        runs = scan_runs()
        done = sum(r["status"] == "done" for r in runs)
        scored = sum(r["test_auc"] is not None for r in runs)
        self.cards["Data"].configure(text=("28 x 28 file downloaded" if npz_path(28).exists() else "not downloaded yet\n(Data tab)"))
        self.cards["Training runs"].configure(text=f"{done} finished\n{len(runs) - done} unfinished folders")
        self.cards["Test results"].configure(text=f"{scored} checkpoints scored" if scored else "not evaluated yet")
        running = self.app.jobs.running()
        self.cards["Jobs running"].configure(text="\n".join(j.title for j in running[:3]) or "none")

        summary = read_csv_rows(RUNS_DIR / "summary.csv")
        rows = []
        for g, label in GROUP_LABELS.items():
            by = {r["row"]: r for r in summary if r["group"] == g}
            if "mean" not in by:
                continue
            m, s, e = by["mean"], by.get("std", {}), by.get("ensemble", {})

            def pm(k: str) -> str:
                return f"{float(m[k]):.3f} ± {float(s[k]):.3f}" if s.get(k) else f"{float(m[k]):.3f}"

            n = sum(1 for k in by if k.startswith("seed"))
            rows.append([label, n, pm("auc"), pm("acc"), pm("sensitivity"), pm("specificity"),
                         f"{float(e['auc']):.3f}" if e else "", f"{float(e['acc']):.3f}" if e else ""])
        self.headline.set_rows(rows)
        compare = results_sections().get("compare", "Run the final models and 'Rebuild results' to see the verdict.")
        set_text(self.verdict, markdown_to_text(compare))


class JobsTab(ttk.Frame):
    """Every job started in this session: status, progress, time left, full output; stop any of them."""

    def __init__(self, master, jobs: JobManager) -> None:
        super().__init__(master, padding=8)
        self.jobs = jobs
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Stop selected", command=self.stop).pack(side="left")
        ttk.Button(bar, text="Stop all", command=self.stop_all).pack(side="left", padx=4)
        self.table = Table(self, ["id", "job", "status", "elapsed", "progress", "time left", "command"],
                           {"id": 40, "job": 230, "command": 420}, height=8)
        self.table.pack(fill="x", pady=6)
        self.table.tree.bind("<<TreeviewSelect>>", lambda _: self.show_log())
        self.log = log_box(self, height=18)
        self.log.pack(fill="both", expand=True)
        jobs.listeners.append(self.on_job)
        self.after(1000, self.refresh)

    def refresh(self) -> None:
        rows, keys = [], []
        for j in reversed(self.jobs.jobs):
            frac, eta = j.progress()
            rows.append([j.id, j.title, j.status, format_seconds(j.elapsed()), f"{frac:.0%}" if frac is not None else "",
                         format_seconds(eta) if j.running and (j.total_epochs or j.progress_fn) else "", j.command()])
            keys.append(str(j.id))
        selected = self.table.selected()
        self.table.set_rows(rows, keys)
        if selected in keys:
            self.table.tree.selection_set(selected)
        self.after(1000, self.refresh)

    def job(self):
        key = self.table.selected()
        return next((j for j in self.jobs.jobs if str(j.id) == key), None)

    def show_log(self) -> None:
        j = self.job()
        if j:
            set_text(self.log, "\n".join(j.lines))
            self.log.see("end")

    def on_job(self, job, new: list[str]) -> None:
        if job is self.job():
            append_log(self.log, new)

    def stop(self) -> None:
        j = self.job()
        if j and j.running and messagebox.askyesno("Stop", f"Stop '{j.title}'?"):
            j.stop()

    def stop_all(self) -> None:
        if self.jobs.running() and messagebox.askyesno("Stop all", "Stop every running job?"):
            self.jobs.stop_all()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Quantum Pneumonia Lab")
        self.geometry("1440x920")
        self.minsize(1100, 700)
        icon = Path(__file__).with_name("icon.png")
        if icon.exists():
            self.iconphoto(True, tk.PhotoImage(file=str(icon)))
        apply_theme(self)

        self.jobs = JobManager(self)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb
        self.tabs = {}
        self.tabs["Overview"] = Overview(nb, self)
        self.tabs["Data"] = DataTab(nb, self.jobs)
        self.tabs["Train"] = TrainTab(nb, self.jobs)
        self.tabs["Runs"] = RunsTab(nb)
        self.tabs["Results"] = ResultsTab(nb, self.jobs)
        self.tabs["Predict"] = PredictTab(nb)
        self.tabs["Learn"] = LearnTab(nb)
        self.tabs["System"] = SystemTab(nb, self.jobs)
        self.tabs["Jobs"] = JobsTab(nb, self.jobs)
        for name, tab in self.tabs.items():
            nb.add(tab, text=name)
        nb.bind("<<NotebookTabChanged>>", lambda _: self.on_tab())

        self.statusbar = ttk.Label(self, text="", foreground=INK2, padding=(8, 2))
        self.statusbar.pack(fill="x")
        self.jobs.listeners.append(lambda job, new: None if job.running else self.tabs["Runs"].refresh())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_status()

    def show(self, name: str) -> None:
        self.nb.select(self.tabs[name])

    def on_tab(self) -> None:
        tab = self.nb.nametowidget(self.nb.select())
        if hasattr(tab, "refresh"):
            tab.refresh()

    def update_status(self) -> None:
        running = self.jobs.running()
        if running:
            parts = []
            for j in running:
                frac, eta = j.progress()
                parts.append(f"{j.title}: {frac:.0%}, {format_seconds(eta)} left" if frac is not None else j.title)
            self.statusbar.configure(text="Running: " + "   |   ".join(parts))
        else:
            self.statusbar.configure(text=f"Ready. Project folder: {ROOT}")
        self.after(1000, self.update_status)

    def on_close(self) -> None:
        running = self.jobs.running()
        if running and not messagebox.askyesno("Jobs running", f"{len(running)} job(s) are still running. "
                                               "Stop them and close?\n\nFinished runs are kept; an interrupted "
                                               "training run can be restarted later."):
            return
        self.jobs.stop_all()
        self.destroy()


if __name__ == "__main__":
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on high-resolution laptop screens
        except (AttributeError, OSError):
            pass
    App().mainloop()
