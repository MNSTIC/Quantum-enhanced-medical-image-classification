"""Predict tab: classify an X-ray, and look inside the model at every step from pixels to probability.

Study project only: this must not be used for diagnosis.
"""
from __future__ import annotations

import csv
import math
import queue
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from common import FONT_BOLD, FONT_TITLE, INK2, RUNS_DIR, SERIES, PlotPanel, Table, section, style_axes
from tab_data import IMAGE_TYPES, custom_sets


def trained_groups() -> list[str]:
    return sorted({p.parent.parent.name for p in RUNS_DIR.glob("*/seed*/best.pt")})


class PredictTab(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=8)
        self.models, self.models_group, self.img, self.truth = None, None, None, None
        self.results: queue.Queue = queue.Queue()
        self.photos = []

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Label(bar, text="Model:").pack(side="left")
        groups = trained_groups()
        default = "hybrid_q8_l1" if "hybrid_q8_l1" in groups else (groups[0] if groups else "")
        self.group = tk.StringVar(value=default)
        ttk.Combobox(bar, textvariable=self.group, values=groups, width=30, state="readonly").pack(side="left", padx=4)
        ttk.Button(bar, text="Open image...", command=self.open_image).pack(side="left", padx=4)
        ttk.Button(bar, text="Random validation image", command=self.random_val).pack(side="left", padx=4)
        ttk.Label(bar, text="  Explain seed:").pack(side="left")
        self.seed = tk.StringVar(value="0")
        ttk.Spinbox(bar, textvariable=self.seed, from_=0, to=9, width=4, command=self.predict).pack(side="left")
        self.status = ttk.Label(bar, text="", foreground=INK2)
        self.status.pack(side="left", padx=8)
        ttk.Label(self, text="Study project only. Not for diagnosis.", foreground="#b3261e", font=FONT_BOLD).pack(anchor="w")

        top = ttk.Frame(self)
        top.pack(fill="x", pady=4)
        imgs = section(top, "Image")
        imgs.pack(side="left", fill="y")
        self.canvas = tk.Canvas(imgs, width=420, height=210, background="#fcfcfb", highlightthickness=0)
        self.canvas.pack()
        verdict = section(top, "Prediction")
        verdict.pack(side="left", fill="both", expand=True, padx=6)
        self.answer = ttk.Label(verdict, text="Open an image or pick a random validation image.", font=FONT_TITLE)
        self.answer.pack(anchor="w")
        self.answer2 = ttk.Label(verdict, text="", foreground=INK2, wraplength=600, justify="left")
        self.answer2.pack(anchor="w")
        self.seed_plot = PlotPanel(verdict, figsize=(6, 1.7), toolbar=False)
        self.seed_plot.pack(fill="both", expand=True)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.explain = PlotPanel(nb, figsize=(11, 3.6))
        nb.add(self.explain, text="Inside the model (one seed)")
        batch = ttk.Frame(nb, padding=6)
        nb.add(batch, text="Predict a whole image set")
        row = ttk.Frame(batch)
        row.pack(fill="x")
        ttk.Label(row, text="Image set (add sets in the Data tab):").pack(side="left")
        self.set_name = tk.StringVar()
        self.set_box = ttk.Combobox(row, textvariable=self.set_name, width=30, state="readonly",
                                    postcommand=lambda: self.set_box.configure(values=[s["name"] for s in custom_sets()]))
        self.set_box.pack(side="left", padx=4)
        ttk.Button(row, text="Predict all", command=self.predict_set).pack(side="left", padx=4)
        ttk.Button(row, text="Save as CSV...", command=self.save_csv).pack(side="left", padx=4)
        self.batch_table = Table(batch, ["file", "P(pneumonia)", "label"], {"file": 420}, height=8)
        self.batch_table.pack(fill="both", expand=True, pady=4)
        self.batch_rows: list[list] = []
        self.after(200, self.poll)

    # ------------------------------------------------------------------------------ inputs
    def open_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images or arrays", " ".join(f"*{t}" for t in IMAGE_TYPES))])
        if path:
            from predict import load_image

            self.img, self.truth = load_image(Path(path)), None
            self.predict()

    def random_val(self) -> None:
        self.status.configure(text="loading the validation set...")

        def work() -> None:
            from PIL import Image

            from data import CLASS_NAMES, load_split

            ds = load_split("val")
            i = random.randrange(len(ds))
            self.results.put(("image", Image.fromarray(ds.imgs[i]),
                              f"Validation image {i}; true label: {CLASS_NAMES[int(ds.labels[i][0])]}."))

        threading.Thread(target=work, daemon=True).start()

    # --------------------------------------------------------------------------- prediction
    def predict(self) -> None:
        if self.img is None or not self.group.get():
            return
        self.status.configure(text="working... (the first prediction loads PyTorch and the models)")
        group, img, seed = self.group.get(), self.img, self.seed.get()

        def work() -> None:
            try:
                from predict import load_models, predict_image

                if self.models_group != group:
                    self.models, self.models_group = load_models(RUNS_DIR / group), group
                probs = predict_image(img, self.models)
                self.results.put(("single", img, probs, self.trace(img, int(seed) if seed.isdigit() else 0)))
            except Exception as e:
                self.results.put(("error", repr(e)))

        threading.Thread(target=work, daemon=True).start()

    def trace(self, img, seed: int) -> dict:
        """Every intermediate quantity for one seed: pixels -> features -> angles -> circuit -> logit."""
        import torch

        from predict import to_tensor

        name, cfg, m = self.models[min(seed, len(self.models) - 1)]
        x = to_tensor(img, cfg.size)
        out = {"name": name, "mode": cfg.mode, "x": x[0, 0].numpy()}
        with torch.no_grad():
            if cfg.mode in ("hybrid", "classical"):
                f = m.backbone(x)
                a = math.pi * torch.tanh(m.pre(m.drop(f)))
                z = m.middle(a).float()
                contrib = m.head.weight[0] * z[0]
                out.update(features=f[0].numpy(), angles=a[0].numpy(), outputs=z[0].numpy(),
                           contrib=contrib.numpy(), bias=float(m.head.bias[0]), logit=float(contrib.sum() + m.head.bias[0]))
            else:
                maps = m.front(x)[0].numpy()
                logit = float(m(x)[0])
                out.update(maps=maps, logit=logit)
        return out

    def poll(self) -> None:
        try:
            while True:
                msg = self.results.get_nowait()
                if msg[0] == "error":
                    self.status.configure(text="")
                    messagebox.showerror("Prediction failed", msg[1])
                elif msg[0] == "image":
                    self.img, self.truth = msg[1], msg[2]
                    self.predict()
                elif msg[0] == "single":
                    self.show(*msg[1:])
                elif msg[0] == "batch":
                    self.batch_rows = msg[1]
                    self.batch_table.set_rows(self.batch_rows)
                    self.status.configure(text=f"{len(self.batch_rows)} images predicted")
        except queue.Empty:
            pass
        self.after(200, self.poll)

    def show(self, img, probs: list[float], tr: dict) -> None:
        from PIL import Image, ImageTk

        from data import CLASS_NAMES

        self.status.configure(text="")
        p = sum(probs) / len(probs)
        self.answer.configure(text=f"P(pneumonia) = {p:.3f}  ->  {CLASS_NAMES[int(p > 0.5)]}")
        self.answer2.configure(text=f"Mean of {len(probs)} model(s) in runs/{self.models_group}, threshold 0.5. "
                                    + (self.truth or "Your image: true label unknown."))
        # image and what the model actually sees
        self.photos = [ImageTk.PhotoImage(img.resize((200, 200))),
                       ImageTk.PhotoImage(Image.fromarray(((tr["x"] * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8"))
                                          .resize((200, 200), Image.NEAREST))]
        self.canvas.delete("all")
        for k, (ph, label) in enumerate(zip(self.photos, ("your image (square crop)", f"model input ({tr['x'].shape[0]}x{tr['x'].shape[0]})"))):
            self.canvas.create_image(5 + 210 * k, 0, anchor="nw", image=ph)
            self.canvas.create_text(5 + 210 * k, 204, anchor="nw", text=label, fill=INK2)
        # per-seed probabilities
        fig = self.seed_plot.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)
        ax.bar(range(len(probs)), probs, color=SERIES[0], width=0.6)
        ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(probs)), [n for n, _, _ in self.models], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(pneumonia)", fontsize=8)
        self.seed_plot.draw()
        self.draw_trace(tr)

    def draw_trace(self, tr: dict) -> None:
        fig = self.explain.figure
        fig.clear()
        if "maps" in tr:
            for k in range(4):
                ax = fig.add_subplot(1, 4, k + 1)
                ax.imshow(tr["maps"][k], cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_title(f"front-end feature map {k + 1}", fontsize=8)
                ax.axis("off")
            fig.suptitle(f"{tr['name']}: the 4 feature maps of the 2x2 front end; final logit {tr['logit']:+.2f}", fontsize=9)
            self.explain.draw()
            return
        steps = [("1. CNN: 64 image features", tr["features"], None),
                 ("2. Linear + pi*tanh: 8 rotation angles", tr["angles"], (-math.pi, math.pi)),
                 ("3. Middle block: 8 outputs" + (" <Z> per qubit" if tr["mode"] == "hybrid" else ""), tr["outputs"], (-1, 1)),
                 ("4. Head: weight x output per unit", tr["contrib"], None)]
        for k, (title, v, lim) in enumerate(steps):
            ax = fig.add_subplot(1, 4, k + 1)
            style_axes(ax)
            ax.bar(range(len(v)), v, color=[SERIES[0] if t >= 0 else SERIES[1] for t in v], width=0.7)
            ax.axhline(0, color=INK2, linewidth=0.8)
            if lim:
                ax.set_ylim(*lim)
            ax.set_title(title, fontsize=8)
        p = 1 / (1 + math.exp(-tr["logit"]))
        fig.suptitle(f"{tr['name']}: logit = sum of step 4 ({sum(tr['contrib']):+.2f}) + bias ({tr['bias']:+.2f}) = "
                     f"{tr['logit']:+.2f};  sigmoid(logit) = P(pneumonia) = {p:.3f}", fontsize=9)
        self.explain.draw()

    # ------------------------------------------------------------------------------- batch
    def predict_set(self) -> None:
        sets = {s["name"]: s for s in custom_sets()}
        s = sets.get(self.set_name.get())
        if not s or not self.group.get():
            messagebox.showinfo("Choose a set", "Pick an image set (add one in the Data tab) and a model.")
            return
        files = sorted(p for p in Path(s["folder"]).rglob("*") if p.suffix.lower() in IMAGE_TYPES)
        group = self.group.get()
        self.status.configure(text=f"predicting {len(files)} images...")

        def work() -> None:
            try:
                from data import CLASS_NAMES
                from predict import load_image, load_models, predict_image

                if self.models_group != group:
                    self.models, self.models_group = load_models(RUNS_DIR / group), group
                rows = []
                for f in files:
                    q = predict_image(load_image(f), self.models)
                    p = sum(q) / len(q)
                    rows.append([str(f.relative_to(s["folder"])), round(p, 4), CLASS_NAMES[int(p > 0.5)]])
                self.results.put(("batch", rows))
            except Exception as e:
                self.results.put(("error", repr(e)))

        threading.Thread(target=work, daemon=True).start()

    def save_csv(self) -> None:
        if not self.batch_rows:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "p_pneumonia", "label"])
                w.writerows(self.batch_rows)
