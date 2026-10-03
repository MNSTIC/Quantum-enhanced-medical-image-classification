"""Train tab: configure and start one training run, or a whole stage of runs, and watch it live."""
from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk

from common import (FONT_BOLD, INK2, SERIES, PlotPanel, Table, append_log, gpu_available, log_box, python_for,
                    read_json, scan_runs, section, style_axes)
from config import MODES, Config
from jobs import Job, JobManager, format_seconds, stage_progress_fn
from sweep import STAGES, to_args

STAGE_INFO = {
    "sweep": "Phase 3: 8 circuit settings x 3 seeds, judged on validation data only",
    "final": "Phase 4: final hybrid, classical control and no-entanglement ablation, 5 seeds each",
    "matched": "Phase 4 secondary: the 72-parameter 8 x 3 circuit, 5 seeds",
    "quanv": "Phase 5.2: quantum 2x2 front end and its convolution control, 5 seeds each",
    "s64": "Phase 5.3: final hybrid and control on 64 x 64 images, 5 seeds each",
    "phase7": "Phase 7: 60 epochs + weight averaging (EMA), hybrid and control, 5 seeds each",
}

# (label, Config field, kind, options)
FORM = [
    ("Model", "mode", "choice", MODES),
    ("Qubits (or units)", "n_qubits", "int", (2, 10)),
    ("Circuit layers", "n_layers", "int", (1, 6)),
    ("Entanglement (CNOT ring)", "entangle", "bool", None),
    ("Data re-uploading", "reupload", "bool", None),
    ("Quanvolution layers", "quanv_layers", "int", (1, 4)),
    ("Image size", "size", "choice", (28, 64, 128)),
    ("Epochs", "epochs", "int", (1, 300)),
    ("Seed", "seed", "int", (0, 999)),
    ("Dropout", "dropout", "float", None),
    ("Learning rate, classical", "lr_classical", "float", None),
    ("Learning rate, circuit", "lr_quantum", "float", None),
    ("Weight averaging (EMA decay, 0 = off)", "ema", "float", None),
    ("CPU threads", "threads", "int", (1, 32)),
    ("Device", "device", "choice", ("cpu", "cuda") if gpu_available() else ("cpu",)),
]


class TrainTab(ttk.Frame):
    def __init__(self, master, jobs: JobManager) -> None:
        super().__init__(master, padding=8)
        self.jobs, self.job, self.vars = jobs, None, {}
        jobs.listeners.append(self.on_job)

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        # ---- single run form
        form = section(left, "One training run")
        form.pack(fill="x")
        defaults = Config(mode="hybrid", n_layers=1)   # the final hybrid, as a sensible starting point
        for r, (label, name, kind, opts) in enumerate(FORM):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=1)
            value = getattr(defaults, name)
            if kind == "bool":
                var = tk.BooleanVar(value=value)
                w = ttk.Checkbutton(form, variable=var)
            elif kind == "choice":
                var = tk.StringVar(value=str(value))
                w = ttk.Combobox(form, textvariable=var, values=[str(o) for o in opts], width=14, state="readonly")
            elif kind == "int":
                var = tk.StringVar(value=str(value))
                w = ttk.Spinbox(form, textvariable=var, from_=opts[0], to=opts[1], width=14)
            else:
                var = tk.StringVar(value=str(value))
                w = ttk.Entry(form, textvariable=var, width=16)
            w.grid(row=r, column=1, sticky="w", pady=1)
            var.trace_add("write", lambda *_: self.update_preview())
            self.vars[name] = var
        self.force = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Retrain if it already exists", variable=self.force,
                        command=self.update_preview).grid(row=len(FORM), column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.preview = ttk.Label(form, text="", foreground=INK2, wraplength=330, justify="left")
        self.preview.grid(row=len(FORM) + 1, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(form, text="Start training", command=self.start_run).grid(row=len(FORM) + 2, column=0, sticky="w")

        # ---- stage runner
        st = section(left, "A whole stage (many runs, in parallel)")
        st.pack(fill="x", pady=(8, 0))
        self.stage = tk.StringVar(value="final")
        ttk.Combobox(st, textvariable=self.stage, values=list(STAGES), state="readonly", width=12).grid(row=0, column=0)
        ttk.Label(st, text="workers").grid(row=0, column=1, padx=(8, 2))
        self.workers = tk.StringVar(value="8")
        ttk.Spinbox(st, textvariable=self.workers, from_=1, to=16, width=4).grid(row=0, column=2)
        ttk.Button(st, text="Start stage", command=self.start_stage).grid(row=0, column=3, padx=(8, 0))
        self.stage_info = ttk.Label(st, text="", foreground=INK2, wraplength=330, justify="left")
        self.stage_info.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.stage.trace_add("write", lambda *_: self.update_stage_info())
        self.update_stage_info()

        # ---- live view
        top = ttk.Frame(right)
        top.pack(fill="x")
        self.title = ttk.Label(top, text="No job started from this tab yet.", font=FONT_BOLD)
        self.title.pack(side="left")
        ttk.Button(top, text="Stop", command=self.stop).pack(side="right")
        self.bar = ttk.Progressbar(right, mode="determinate", maximum=1.0)
        self.bar.pack(fill="x", pady=4)
        self.detail = ttk.Label(right, text="", foreground=INK2)
        self.detail.pack(anchor="w")

        views = ttk.Notebook(right)
        views.pack(fill="both", expand=True, pady=4)
        self.plot = PlotPanel(views, figsize=(8, 3.6))
        views.add(self.plot, text="Live curves")
        self.stage_table = Table(views, ["run", "status", "epochs", "best val AUC", "val loss", "minutes"],
                                 {"run": 220, "status": 110})
        views.add(self.stage_table, text="Runs in this stage")
        self.log = log_box(right, height=10)
        self.log.pack(fill="both", expand=False)
        self.update_preview()
        self.redraw()
        self.after(2000, self.refresh_stage_table)

    # ------------------------------------------------------------------ configuration and estimates
    def config(self) -> Config:
        """The form's values as a Config (raises ValueError if a number field is not a number)."""
        cast = {"bool": bool, "int": int, "float": float, "choice": str}
        values = {name: cast[kind](self.vars[name].get()) for _, name, kind, _ in FORM}
        values["size"] = int(values["size"])
        return Config(**values)

    def estimate(self, cfg: Config) -> str:
        """Time for this run, from earlier finished runs of the same kind (median seconds per epoch)."""
        same = []
        for r in scan_runs():
            c = read_json(r["path"] / "config.json")
            if r["status"] == "done" and r["epoch_s"] and all(
                    c.get(k, getattr(Config(), k)) == getattr(cfg, k) for k in ("mode", "n_qubits", "n_layers", "size", "device")):
                same.append(r["epoch_s"])
        if not same:
            return "No earlier run of this kind to estimate the time from."
        per = min(same)   # many earlier runs shared the CPU with others; the fastest is closest to a run on its own
        return (f"Estimated time on its own: about {format_seconds(per * cfg.epochs)} ({per:.1f} s per epoch, fastest "
                f"of {len(same)} earlier runs). Slower while other jobs share the CPU.")

    def update_preview(self) -> None:
        try:
            cfg = self.config()
        except ValueError:
            self.preview.configure(text="Check the values: a number field is not a number.")
            return
        done = (cfg.run_dir() / "metrics.json").exists()
        note = ("\nThis run already exists; tick 'Retrain' to overwrite it, or change the seed."
                if done and not self.force.get() else "")
        self.preview.configure(text=f"Saved to runs/{cfg.run_name()}/seed{cfg.seed}/\n{self.estimate(cfg)}{note}")

    def update_stage_info(self) -> None:
        settings, seeds = STAGES[self.stage.get()]
        runs = [replace(s, seed=k) for s in settings for k in seeds]
        done = sum((r.run_dir() / "metrics.json").exists() for r in runs)
        self.stage_info.configure(text=f"{STAGE_INFO.get(self.stage.get(), '')}\n{len(runs)} runs, {done} already "
                                       f"finished (finished runs are skipped).")

    # ------------------------------------------------------------------------------- starting
    def start_run(self) -> None:
        try:
            cfg = self.config()
        except ValueError:
            messagebox.showerror("Invalid value", "One of the number fields is not a number.")
            return
        if (cfg.run_dir() / "metrics.json").exists() and not self.force.get():
            messagebox.showinfo("Already trained", "This run already exists. Tick 'Retrain' or change the seed.")
            return
        args = ["src/train.py", *to_args(cfg)] + (["--force"] if self.force.get() else [])
        self.attach(self.jobs.submit(Job(f"train {cfg.run_name()}/seed{cfg.seed}", python_for(cfg.device), args,
                                         total_epochs=cfg.epochs)))

    def start_stage(self) -> None:
        stage = self.stage.get()
        args = ["src/sweep.py", "--stage", stage, "--workers", self.workers.get()]
        self.attach(self.jobs.submit(Job(f"stage {stage}", python_for("cpu"), args,
                                         progress_fn=stage_progress_fn([stage]))))

    def attach(self, job: Job) -> None:
        self.job = job
        self.title.configure(text=job.title)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.redraw()

    def stop(self) -> None:
        if self.job and self.job.running and messagebox.askyesno("Stop", f"Stop '{self.job.title}'?"):
            self.job.stop()

    # ------------------------------------------------------------------------------- live view
    def on_job(self, job: Job, new: list[str]) -> None:
        if job is not self.job:
            return
        append_log(self.log, new)
        frac, eta = job.progress()
        self.bar["value"] = frac or 0
        parts = [job.status, f"elapsed {format_seconds(job.elapsed())}"]
        if job.running:
            parts.append(f"time left {format_seconds(eta)}")
        if job.epochs:
            e = job.epochs[-1]
            best = max(x["val_auc"] for x in job.epochs)
            parts.append(f"epoch {int(e['epoch'])}/{job.total_epochs}  val AUC {e['val_auc']:.4f} (best {best:.4f})")
        elif frac is not None and job.progress_fn:
            parts.append(f"{frac:.0%} of all epochs")
        self.detail.configure(text="  |  ".join(parts))
        if any(" ep " in f" {line}" for line in new) or not job.running:
            self.redraw()
        if not job.running:
            self.update_preview()
            self.update_stage_info()

    def redraw(self) -> None:
        fig = self.plot.figure
        fig.clear()
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        for ax in (ax1, ax2):
            style_axes(ax)
            ax.set_xlabel("epoch", fontsize=8)
        ep = self.job.epochs if self.job else []
        if ep:
            x = [e["epoch"] for e in ep]
            ax1.plot(x, [e["train_loss"] for e in ep], color=SERIES[0], label="training loss")
            ax1.plot(x, [e["val_loss"] for e in ep], color=SERIES[1], label="validation loss")
            ax2.plot(x, [e["val_auc"] for e in ep], color=SERIES[2], label="validation AUC", marker="o", markersize=3)
            ax1.legend(fontsize=8, frameon=False)
            ax2.legend(fontsize=8, frameon=False)
        ax1.set_title("Loss", fontsize=9)
        ax2.set_title("Validation AUC (picks the checkpoint)", fontsize=9)
        self.plot.draw()

    def refresh_stage_table(self) -> None:
        if self.job and self.job.progress_fn:
            stage = self.job.args[self.job.args.index("--stage") + 1]
            settings, seeds = STAGES[stage]
            rows, keys = [], []
            for cfg in (replace(s, seed=k) for s in settings for k in seeds):
                d = cfg.run_dir()
                m = read_json(d / "metrics.json")
                hist = sum(1 for _ in open(d / "history.csv", encoding="utf-8")) - 1 if (d / "history.csv").exists() else 0
                status = "done" if m else ("running" if hist else "waiting")
                rows.append([f"{cfg.run_name()}/seed{cfg.seed}", status, f"{cfg.epochs if m else hist}/{cfg.epochs}",
                             m.get("val_auc"), m.get("val_loss"), m.get("train_s", 0) / 60 if m else None])
                keys.append(str(d))
            self.stage_table.set_rows(rows, keys)
        self.after(3000, self.refresh_stage_table)
