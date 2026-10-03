"""Results tab: every generated results table and figure, and buttons that regenerate them."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, ttk

from common import (FIG_DIR, FONT_MONO, INK2, ROOT, append_log, log_box, markdown_to_text, python_for,
                    results_sections, set_text)
from jobs import Job, JobManager, format_seconds

SECTION_NAMES = {
    "main": "Main result: hybrid vs classical (test)", "compare": "Did the quantum layer help? (verdicts)",
    "target": "Target vs published ResNet-18", "seeds": "Per-seed scores", "baselines": "Published baselines",
    "matched": "Parameter-matched 8x3 circuit", "threshold": "Validation-tuned threshold (secondary)",
    "sweep": "Phase 3 sweep (validation only)", "gradients": "Barren-plateau check (gradients)",
    "noise": "Noise and shots: final 1-layer hybrid", "noise_matched": "Noise and shots: 3-layer hybrid",
    "quanv": "Quanvolution front end", "s64": "64 x 64 images", "phase7": "Phase 7: improved recipe",
    "phase7_val": "Phase 7: validation check", "devices": "CPU vs GPU timing", "timing": "Training time",
}


class ResultsTab(ttk.Frame):
    def __init__(self, master, jobs: JobManager) -> None:
        super().__init__(master, padding=8)
        self.jobs, self.job, self.image = jobs, None, None
        jobs.listeners.append(self.on_job)

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="Rebuild tables and figures", command=lambda: self.run([])).pack(side="left")
        ttk.Button(bar, text="... plus noise study (slow)", command=lambda: self.run(["--noise"])).pack(side="left", padx=4)
        ttk.Button(bar, text="Open REPORT.md", command=lambda: os.startfile(ROOT / "REPORT.md")).pack(side="left", padx=4)
        ttk.Button(bar, text="Open figures folder", command=lambda: os.startfile(FIG_DIR)).pack(side="left", padx=4)
        self.status = ttk.Label(bar, text="", foreground=INK2)
        self.status.pack(side="left", padx=8)
        ttk.Label(self, foreground=INK2, wraplength=1100, justify="left",
                  text="evaluate.py is the only program that reads the test set. Every checkpoint is scored once and "
                       "cached, so rebuilding only redraws tables and figures; it never changes a result.").pack(anchor="w", pady=4)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        # tables
        tables = ttk.Frame(nb)
        nb.add(tables, text="Tables")
        self.sections = tk.Listbox(tables, width=38, exportselection=False, font=("Segoe UI", 10))
        self.sections.pack(side="left", fill="y")
        self.sections.bind("<<ListboxSelect>>", lambda _: self.show_section())
        self.text = tk.Text(tables, font=FONT_MONO, wrap="none", state="disabled")
        ys = ttk.Scrollbar(tables, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(tables, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        xs.pack(side="bottom", fill="x")
        self.text.pack(side="left", fill="both", expand=True, padx=(6, 0))
        ys.pack(side="left", fill="y")
        # figures
        figs = ttk.Frame(nb)
        nb.add(figs, text="Figures")
        self.figlist = tk.Listbox(figs, width=38, exportselection=False, font=("Segoe UI", 10))
        self.figlist.pack(side="left", fill="y")
        self.figlist.bind("<<ListboxSelect>>", lambda _: self.show_figure())
        self.canvas = tk.Canvas(figs, background="#fcfcfb", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.canvas.bind("<Configure>", lambda _: self.show_figure())
        self.canvas.bind("<Double-Button-1>", lambda _: self.open_figure())
        # log
        logf = ttk.Frame(nb)
        nb.add(logf, text="Log")
        self.log = log_box(logf, height=20)
        self.log.pack(fill="both", expand=True)
        self.reload()

    def reload(self) -> None:
        self.data = results_sections()
        self.keys = list(self.data)
        self.sections.delete(0, "end")
        for k in self.keys:
            self.sections.insert("end", SECTION_NAMES.get(k, k))
        self.figs = sorted(FIG_DIR.glob("*.png"))
        self.figlist.delete(0, "end")
        for f in self.figs:
            self.figlist.insert("end", f.stem.replace("_", " "))
        if self.keys:
            self.sections.selection_set(0)
            self.show_section()
        if not self.data:
            set_text(self.text, "No results yet. Train the final models, then press 'Rebuild tables and figures'.")

    def show_section(self) -> None:
        sel = self.sections.curselection()
        if sel:
            set_text(self.text, markdown_to_text(self.data[self.keys[sel[0]]]))

    def show_figure(self) -> None:
        sel = self.figlist.curselection()
        if not sel:
            return
        from PIL import Image, ImageTk

        img = Image.open(self.figs[sel[0]])
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        img.thumbnail((w, h), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(img)   # keep a reference, or Tk shows nothing
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.image)
        self.canvas.create_text(8, h - 8, anchor="sw", fill=INK2, text="double-click to open full size")

    def open_figure(self) -> None:
        sel = self.figlist.curselection()
        if sel:
            os.startfile(self.figs[sel[0]])

    def run(self, extra: list[str]) -> None:
        if self.job and self.job.running:
            messagebox.showinfo("Busy", "evaluate.py is already running.")
            return
        if extra and not messagebox.askyesno("Noise study", "The noise study simulates density matrices and can take "
                                             "an hour or more. Already computed settings are reused. Start it?"):
            return
        self.job = self.jobs.submit(Job("evaluate " + " ".join(extra), python_for("cpu"), ["src/evaluate.py", *extra]))

    def on_job(self, job: Job, new: list[str]) -> None:
        if job is not self.job:
            return
        append_log(self.log, new)
        self.status.configure(text=f"evaluate.py: {job.status}, {format_seconds(job.elapsed())}")
        if not job.running:
            self.reload()
