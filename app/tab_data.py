"""Data tab: the dataset files on disk, their class counts, an image browser, and your own image sets.

"Add / remove dataset" here means two things:
  * PneumoniaMNIST at each resolution (28, 64, 128, 224 px) can be downloaded, verified or deleted;
  * folders of your own X-rays can be registered as image sets for batch prediction (Predict tab).
The training data itself never changes: the experiment is defined on the official MedMNIST splits.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from common import DATA_DIR, FONT_BOLD, INK2, Table, python_for, section
from jobs import Job, JobManager

SIZES = (28, 64, 128, 224)
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
SETS_FILE = DATA_DIR / "custom_sets.json"


def npz_path(size: int) -> Path:
    return DATA_DIR / ("pneumoniamnist.npz" if size == 28 else f"pneumoniamnist_{size}.npz")


def custom_sets() -> list[dict]:
    try:
        return json.loads(SETS_FILE.read_text(encoding="utf-8"))["sets"]
    except (OSError, ValueError, KeyError):
        return []


def save_sets(sets: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    SETS_FILE.write_text(json.dumps({"sets": sets}, indent=2), encoding="utf-8")


class DataTab(ttk.Frame):
    def __init__(self, master, jobs: JobManager) -> None:
        super().__init__(master, padding=8)
        self.jobs, self.job = jobs, None
        self.q: queue.Queue = queue.Queue()
        self.arrays: dict = {}
        self.photos: list = []
        jobs.listeners.append(self.on_job)

        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        files = section(left, "PneumoniaMNIST files (MedMNIST v2, CC BY 4.0)")
        files.pack(fill="x")
        self.files = Table(files, ["resolution", "file", "status", "size (MB)"], {"file": 190, "status": 160}, height=4)
        self.files.pack(fill="x")
        row = ttk.Frame(files)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Download", command=self.download).pack(side="left")
        ttk.Button(row, text="Verify (MD5 + class counts)", command=self.verify).pack(side="left", padx=4)
        ttk.Button(row, text="Delete file...", command=self.delete).pack(side="left")
        ttk.Button(row, text="Open data folder", command=lambda: os.startfile(DATA_DIR)).pack(side="left", padx=4)
        self.files_note = ttk.Label(files, text="", foreground=INK2, wraplength=560, justify="left")
        self.files_note.pack(anchor="w")
        self.files.tree.bind("<<TreeviewSelect>>", lambda _: self.load_stats())

        stats = section(left, "Class counts (selected resolution)")
        stats.pack(fill="x", pady=8)
        self.stats = Table(stats, ["split", "images", "normal", "pneumonia", "share"], height=3)
        self.stats.pack(fill="x")
        ttk.Label(stats, foreground=INK2, wraplength=560, justify="left",
                  text="Test counts are the documented ones (624 = 234 normal + 390 pneumonia); the app never "
                       "opens the test images. Test has proportionally fewer pneumonia cases than train and "
                       "validation, which is why test scores sit below validation scores.").pack(anchor="w", pady=(4, 0))

        sets = section(left, "Your own image sets (for batch prediction)")
        sets.pack(fill="both", expand=True)
        self.sets = Table(sets, ["name", "folder", "images"], {"folder": 330}, height=5)
        self.sets.pack(fill="both", expand=True)
        row = ttk.Frame(sets)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Add folder...", command=self.add_set).pack(side="left")
        ttk.Button(row, text="Remove from list", command=self.remove_set).pack(side="left", padx=4)
        ttk.Label(sets, foreground=INK2, text="Removing only forgets the folder; your files are not touched.").pack(anchor="w")

        browse = section(right, "Image browser")
        browse.pack(fill="both", expand=True)
        row = ttk.Frame(browse)
        row.pack(fill="x")
        self.split = tk.StringVar(value="train")
        ttk.Combobox(row, textvariable=self.split, values=("train", "val"), width=7, state="readonly").pack(side="left")
        self.cls = tk.StringVar(value="any class")
        ttk.Combobox(row, textvariable=self.cls, values=("any class", "normal", "pneumonia"), width=11,
                     state="readonly").pack(side="left", padx=4)
        ttk.Button(row, text="Random image", command=self.random_image).pack(side="left")
        ttk.Button(row, text="8 augmentations", command=self.augment).pack(side="left", padx=4)
        self.caption = ttk.Label(browse, text="", font=FONT_BOLD)
        self.caption.pack(anchor="w", pady=4)
        self.canvas = tk.Canvas(browse, background="#fcfcfb", highlightthickness=0, height=420)
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(browse, foreground=INK2, wraplength=520, justify="left",
                  text="Training adds small random rotations (up to 10 degrees), shifts (8%) and zooms (0.9-1.1). "
                       "Never flips: a mirrored chest would put the heart on the wrong side.").pack(anchor="w")

        self.refresh_files()
        self.refresh_sets()
        self.after(200, self.poll)

    def refresh(self) -> None:
        """Called whenever the tab is opened."""
        self.refresh_files()
        self.refresh_sets()
        if not self.photos and npz_path(28).exists():
            self.random_image()

    # -------------------------------------------------------------------------------- files
    def refresh_files(self) -> None:
        rows = []
        for s in SIZES:
            p = npz_path(s)
            rows.append([f"{s} x {s}", p.name, "downloaded" if p.exists() else "not downloaded",
                         f"{p.stat().st_size / 2 ** 20:.1f}" if p.exists() else ""])
        self.files.set_rows(rows, keys=[str(s) for s in SIZES])
        if not self.files.selected():
            self.files.tree.selection_set("28")

    def size(self) -> int:
        return int(self.files.selected() or 28)

    def download(self) -> None:
        if self.job and self.job.running:
            return
        s = self.size()
        self.files_note.configure(text=f"Downloading the {s} x {s} file from Zenodo (MD5-checked by medmnist)...")
        self.job = self.jobs.submit(Job(f"download data {s}px", python_for("cpu"), ["src/data.py", "--size", str(s)]))

    def on_job(self, job: Job, new: list[str]) -> None:
        if job is self.job and not job.running:
            ok = job.returncode == 0
            self.files_note.configure(text=("Done: class counts checked." if ok else "Failed:\n" + "\n".join(list(job.lines)[-4:])))
            self.refresh_files()
            self.load_stats()

    def verify(self) -> None:
        s = self.size()
        if not npz_path(s).exists():
            return
        self.files_note.configure(text="Verifying...")

        def work() -> None:
            from medmnist import INFO

            from data import load_split

            info = INFO["pneumoniamnist"]
            expected = info["MD5"] if s == 28 else info.get(f"MD5_{s}")
            md5 = hashlib.md5(npz_path(s).read_bytes()).hexdigest()
            try:
                for split in ("train", "val"):
                    load_split(split, size=s)   # raises if the class counts are not the documented ones
                counts = "class counts as documented"
            except AssertionError as e:
                counts = f"CLASS COUNTS WRONG: {e}"
            self.q.put(("note", f"MD5 {'matches' if md5 == expected else 'DOES NOT MATCH'} the published checksum; {counts}."))

        threading.Thread(target=work, daemon=True).start()

    def delete(self) -> None:
        s = self.size()
        p = npz_path(s)
        if p.exists() and messagebox.askyesno("Delete dataset file", f"Delete {p.name}?\n\nIt can be downloaded again "
                                              "later. Training at this resolution needs it."):
            p.unlink()
            self.arrays.pop(s, None)
            self.refresh_files()
            self.load_stats()

    def load_stats(self) -> None:
        s = self.size()
        if not npz_path(s).exists():
            self.stats.set_rows([])
            return
        import numpy as np

        if s not in self.arrays:
            with np.load(npz_path(s)) as z:   # only the train and validation arrays are read
                self.arrays[s] = {k: z[k] for k in ("train_images", "train_labels", "val_images", "val_labels")}
        a = self.arrays[s]
        rows = []
        for split in ("train", "val"):
            y = a[f"{split}_labels"].reshape(-1)
            rows.append([split, len(y), int((y == 0).sum()), int((y == 1).sum()), f"{y.mean():.1%}"])
        rows.append(["test", 624, 234, 390, f"{390 / 624:.1%}"])
        self.stats.set_rows(rows)

    # ----------------------------------------------------------------------------- browser
    def random_image(self) -> None:
        from PIL import Image, ImageTk

        s = self.size() if npz_path(self.size()).exists() else 28
        if not npz_path(s).exists():
            messagebox.showinfo("No data", "Download the 28 x 28 file first.")
            return
        self.load_stats()
        a = self.arrays[s]
        y = a[f"{self.split.get()}_labels"].reshape(-1)
        wanted = {"normal": 0, "pneumonia": 1}.get(self.cls.get())
        idx = [i for i in range(len(y)) if wanted is None or y[i] == wanted]
        self.current = (s, self.split.get(), random.choice(idx))
        _, split, i = self.current
        img = Image.fromarray(a[f"{split}_images"][i])
        self.caption.configure(text=f"{split} image {i} ({s} x {s}): {['normal', 'pneumonia'][int(y[i])]}")
        self.photos = [ImageTk.PhotoImage(img.resize((400, 400), Image.NEAREST))]
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photos[0])

    def augment(self) -> None:
        if not getattr(self, "current", None):
            self.random_image()
        s, split, i = self.current
        img_array = self.arrays[s][f"{split}_images"][i]

        def work() -> None:
            from PIL import Image

            from data import make_transform

            tf = make_transform(train=True)
            outs = [(tf(Image.fromarray(img_array))[0].numpy() * 0.5 + 0.5) for _ in range(8)]
            self.q.put(("aug", outs))

        threading.Thread(target=work, daemon=True).start()

    def poll(self) -> None:
        from PIL import Image, ImageTk

        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "note":
                    self.files_note.configure(text=payload)
                elif kind == "aug":
                    self.photos = [ImageTk.PhotoImage(Image.fromarray((p.clip(0, 1) * 255).astype("uint8"))
                                                      .resize((130, 130), Image.NEAREST)) for p in payload]
                    self.canvas.delete("all")
                    for k, ph in enumerate(self.photos):
                        self.canvas.create_image(5 + 135 * (k % 4), 5 + 140 * (k // 4), anchor="nw", image=ph)
                    self.caption.configure(text=self.caption.cget("text").split(" -")[0] + " - 8 random training augmentations")
        except queue.Empty:
            pass
        self.after(200, self.poll)

    # -------------------------------------------------------------------------- image sets
    def refresh_sets(self) -> None:
        rows = []
        for s in custom_sets():
            folder = Path(s["folder"])
            n = sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_TYPES) if folder.exists() else "folder missing"
            rows.append([s["name"], s["folder"], n])
        self.sets.set_rows(rows, keys=[r[0] for r in rows])

    def add_set(self) -> None:
        folder = filedialog.askdirectory(title="Folder with X-ray images (PNG, JPG, BMP, TIF or .npy)")
        if not folder:
            return
        name = simpledialog.askstring("Name", "A short name for this image set:", initialvalue=Path(folder).name)
        if not name:
            return
        sets = [s for s in custom_sets() if s["name"] != name] + [{"name": name, "folder": folder}]
        save_sets(sets)
        self.refresh_sets()

    def remove_set(self) -> None:
        name = self.sets.selected()
        if name:
            save_sets([s for s in custom_sets() if s["name"] != name])
            self.refresh_sets()
