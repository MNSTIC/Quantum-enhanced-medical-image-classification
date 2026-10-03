"""Runs tab: every run in runs/, its learning curves, configuration and scores."""
from __future__ import annotations

import json
import os
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from common import (FONT_MONO, INK2, RUNS_DIR, SERIES, PlotPanel, ScrollText, Table, read_csv_rows, read_json,
                    scan_runs, set_text, style_axes)

COLUMNS = ["run", "status", "epochs", "best epoch", "val AUC", "val loss", "test AUC", "test ACC", "minutes", "device"]


class RunsTab(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=8)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Label(bar, text="Filter:").pack(side="left")
        self.filter = tk.StringVar(value="")
        e = ttk.Entry(bar, textvariable=self.filter, width=24)
        e.pack(side="left", padx=4)
        e.bind("<KeyRelease>", lambda _: self.refresh())
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="Delete run...", command=self.delete).pack(side="left", padx=4)
        self.count = ttk.Label(bar, text="", foreground=INK2)
        self.count.pack(side="right")

        panes = ttk.PanedWindow(self, orient="vertical")
        panes.pack(fill="both", expand=True, pady=6)
        self.table = Table(panes, COLUMNS, {"run": 300, "status": 90, "epochs": 70, "best epoch": 95}, height=10)
        panes.add(self.table, weight=1)
        self.table.tree.bind("<<TreeviewSelect>>", lambda _: self.show())

        bottom = ttk.Frame(panes)
        panes.add(bottom, weight=2)
        self.info = ScrollText(bottom, width=44, font=FONT_MONO, wrap="none", state="disabled")
        self.info.pack(side="right", fill="y", padx=(6, 0))   # packed first, so the plot cannot squeeze it
        self.plot = PlotPanel(bottom, figsize=(9, 3.4))
        self.plot.pack(side="left", fill="both", expand=True)
        self.refresh()

    def refresh(self) -> None:
        f = self.filter.get().strip().lower()
        rows = [r for r in scan_runs() if f in f"{r['group']}/{r['seed']}".lower()]
        self.table.set_rows([[f"{r['group']}/{r['seed']}", r["status"], f"{r['epochs_done']}/{r['epochs']}",
                              r["best_epoch"], r["val_auc"], r["val_loss"], r["test_auc"], r["test_acc"],
                              f"{r['train_min']:.1f}" if r["train_min"] else "", r["device"]] for r in rows],
                            keys=[str(r["path"]) for r in rows])
        done = sum(r["status"] == "done" for r in rows)
        self.count.configure(text=f"{len(rows)} runs shown, {done} finished. Test columns appear after evaluate.py.")

    def show(self) -> None:
        key = self.table.selected()
        if not key:
            return
        d = Path(key)
        hist = read_csv_rows(d / "history.csv")
        fig = self.plot.figure
        fig.clear()
        axes = [fig.add_subplot(1, 3, i) for i in (1, 2, 3)]
        for ax in axes:
            style_axes(ax)
            ax.set_xlabel("epoch", fontsize=8)
        if hist:
            ep = [int(h["epoch"]) for h in hist]
            col = lambda k: [float(h[k]) for h in hist]
            axes[0].plot(ep, col("train_loss"), color=SERIES[0], label="training")
            axes[0].plot(ep, col("val_loss"), color=SERIES[1], label="validation")
            axes[0].set_title("Loss", fontsize=9)
            axes[0].legend(fontsize=7, frameon=False)
            axes[1].plot(ep, col("val_auc"), color=SERIES[2], label="val AUC")
            axes[1].plot(ep, col("val_acc"), color=SERIES[3], label="val ACC")
            best = read_json(d / "metrics.json").get("best_epoch")
            if best:
                axes[1].axvline(best, color=INK2, linestyle="--", linewidth=1, label=f"checkpoint (epoch {best})")
            axes[1].set_title("Validation AUC and accuracy", fontsize=9)
            axes[1].legend(fontsize=7, frameon=False)
            axes[2].plot(ep, col("probe_grad_norm"), color=SERIES[0])
            axes[2].set_yscale("log")
            axes[2].set_title("Gradient norm (compared block)", fontsize=9)
        self.plot.draw()
        info = {"config": {k: v for k, v in read_json(d / "config.json").items() if k != "versions"},
                "validation (best epoch)": read_json(d / "metrics.json"),
                "test (after evaluate.py)": read_json(d / "test_metrics.json")}
        set_text(self.info, json.dumps(info, indent=1, default=str))

    def open_folder(self) -> None:
        os.startfile(self.table.selected() or str(RUNS_DIR))

    def delete(self) -> None:
        key = self.table.selected()
        if not key:
            return
        if messagebox.askyesno("Delete run", f"Permanently delete this run folder?\n\n{key}\n\n"
                                             "Its checkpoint, history and predictions are removed. This cannot be undone."):
            shutil.rmtree(key, ignore_errors=True)
            self.refresh()
