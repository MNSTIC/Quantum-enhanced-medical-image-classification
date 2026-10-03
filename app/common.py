"""Shared pieces of the desktop app: project paths, the run index, and small reusable widgets.

The app is a window onto the project, not a copy of it: it imports the project's own code from
src/ and runs the project's own scripts (train.py, sweep.py, evaluate.py, ...) for anything heavy.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import textwrap
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import DATA_DIR, FIG_DIR, RUNS_DIR  # noqa: E402  (config.py imports only the standard library)

PY = ROOT / ".venv" / "Scripts" / "python.exe"          # the project's environment (CPU PyTorch)
PY_GPU = ROOT / ".venv-gpu" / "Scripts" / "python.exe"  # optional environment with CUDA PyTorch

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # one fixed colour per series, as in figures.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_MONO = ("Consolas", 10)


BG, PANEL, EDGE, TAB = "#f6f5f2", "#fcfcfb", "#d6d4cf", "#e9e7e2"


def apply_theme(root: tk.Tk) -> None:
    """One explicit light theme for every widget, so the app looks the same whatever the Windows colour
    settings are, and matches the light figures."""
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=INK, fieldbackground=PANEL, font=FONT, bordercolor=EDGE,
                    lightcolor=BG, darkcolor=EDGE, troughcolor=TAB, selectbackground=SERIES[0], selectforeground="white")
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 6), background=TAB)
    style.map("TNotebook.Tab", background=[("selected", PANEL)])
    style.configure("TButton", padding=(10, 4), background=TAB)
    style.map("TButton", background=[("active", EDGE)])
    style.configure("TLabelframe", background=BG)
    style.configure("TLabelframe.Label", background=BG, foreground=INK, font=FONT_BOLD)
    # Row height from the font's real line height, so letters like g, y and _ are not clipped at any screen scaling.
    line = tkfont.Font(root=root, font=FONT).metrics("linespace")
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=INK, rowheight=int(line * 1.35))
    style.configure("Treeview.Heading", background=TAB, font=FONT_BOLD)
    style.map("Treeview", background=[("selected", SERIES[0])], foreground=[("selected", "white")])
    style.configure("Horizontal.TProgressbar", background=SERIES[0], troughcolor=TAB)
    style.configure("Horizontal.TScale", background=SERIES[0], troughcolor=EDGE)
    root.configure(background=BG)
    # Plain Tk widgets (e.g. matplotlib's toolbar) take their colours from these defaults when created;
    # the toolbar also picks dark or light icons from its background, so this must happen first.
    for widget in ("Frame", "Button", "Checkbutton", "Label"):
        root.option_add(f"*{widget}.background", BG)
        root.option_add(f"*{widget}.foreground", INK)
    for widget in ("Text", "Listbox", "Canvas"):
        root.option_add(f"*{widget}.background", PANEL)
    for widget in ("Text", "Listbox"):
        root.option_add(f"*{widget}.foreground", INK)
        root.option_add(f"*{widget}.selectBackground", SERIES[0])
        root.option_add(f"*{widget}.selectForeground", "white")
    root.option_add("*Font", FONT)


def python_for(device: str = "cpu") -> Path:
    """Interpreter used to run a job: the GPU environment for CUDA jobs, else the project's one."""
    if device == "cuda" and PY_GPU.exists():
        return PY_GPU
    return PY if PY.exists() else Path(sys.executable)


def gpu_available() -> bool:
    return PY_GPU.exists()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def scan_runs() -> list[dict]:
    """One row per runs/<group>/seed<k>/ folder with whatever that run has produced so far."""
    rows = []
    for d in sorted(RUNS_DIR.glob("*/seed*")):
        if not d.is_dir():
            continue
        cfg, m, t = read_json(d / "config.json"), read_json(d / "metrics.json"), read_json(d / "test_metrics.json")
        done = (d / "metrics.json").exists()
        epochs_done = max(count_lines(d / "history.csv") - 1, 0)
        rows.append({
            "group": d.parent.name, "seed": d.name, "path": d,
            "status": "done" if done else ("incomplete" if cfg else "empty"),
            "epochs": cfg.get("epochs"), "epochs_done": cfg.get("epochs") if done else epochs_done,
            "device": cfg.get("device", "cpu"), "mode": cfg.get("mode"),
            "best_epoch": m.get("best_epoch"), "val_auc": m.get("val_auc"), "val_loss": m.get("val_loss"),
            "val_acc": m.get("val_acc"), "train_min": m.get("train_s", 0) / 60 if m else None,
            "epoch_s": m.get("epoch_s_mean"), "n_params": m.get("n_params"),
            "test_auc": t.get("auc"), "test_acc": t.get("acc"), "has_ckpt": (d / "best.pt").exists(),
        })
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def results_sections() -> dict[str, str]:
    """runs/results.md split into {section name: markdown body}."""
    path = RUNS_DIR / "results.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def markdown_to_text(md: str) -> str:
    """Markdown tables -> aligned plain-text tables (everything else is kept as it is)."""
    out, block = [], []

    def flush() -> None:
        if not block:
            return
        rows = [[c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
                for line in block if not re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())]
        widths = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(max(map(len, rows)))]
        for k, r in enumerate(rows):
            out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
            if k == 0:
                out.append("  ".join("-" * w for w in widths))
        block.clear()

    for line in md.replace("−", "-").splitlines():   # U+2212 is missing from Consolas and breaks the line height
        if line.strip().startswith("|"):
            block.append(line)
        else:
            flush()
            out.append(textwrap.fill(line.replace("**", ""), 120) if line.strip() else "")
    flush()
    return "\n".join(out)


def fmt(v, digits: int = 4) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


# ------------------------------------------------------------------------------------- widgets
class Table(ttk.Frame):
    """A sortable table (ttk.Treeview with scrollbars). Click a column header to sort by it."""

    def __init__(self, master, columns: list[str], widths: dict[str, int] | None = None, height: int = 12) -> None:
        super().__init__(master)
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=height, selectmode="browse")
        for c in columns:
            self.tree.heading(c, text=c, command=lambda c=c: self.sort(c))
            self.tree.column(c, width=(widths or {}).get(c, 100), anchor="w", stretch=True)
        ys = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._reverse: dict[str, bool] = {}

    def set_rows(self, rows: list[list], keys: list[str] | None = None) -> None:
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(rows):
            self.tree.insert("", "end", iid=keys[i] if keys else None, values=[fmt(v) for v in r])

    def selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def sort(self, col: str) -> None:
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]

        def key(item):
            try:
                return (0, float(item[0]))
            except ValueError:
                return (1, item[0])

        rev = self._reverse.get(col, False)
        for i, (_, k) in enumerate(sorted(items, key=key, reverse=rev)):
            self.tree.move(k, "", i)
        self._reverse[col] = not rev


class PlotPanel(ttk.Frame):
    """A matplotlib figure inside the window, with the usual zoom/pan/save toolbar."""

    def __init__(self, master, figsize=(7, 4), toolbar: bool = True) -> None:
        super().__init__(master)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        # "constrained" layout re-fits titles, labels and ticks on every redraw, including window resizes
        self.figure = Figure(figsize=figsize, dpi=100, facecolor=PANEL, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        if toolbar:
            NavigationToolbar2Tk(self.canvas, self, pack_toolbar=True).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self) -> None:
        self.canvas.draw_idle()


def style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.set_axisbelow(True)   # grid lines behind the bars and lines, not over them
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


class ScrollText(tk.Text):
    """A Text with a themed vertical scrollbar. Like tkinter's ScrolledText, packing it packs its frame."""

    def __init__(self, master, **kw) -> None:
        self.frame = ttk.Frame(master)
        bar = ttk.Scrollbar(self.frame, orient="vertical")
        bar.pack(side="right", fill="y")
        super().__init__(self.frame, yscrollcommand=bar.set, **kw)
        bar.configure(command=self.yview)
        super().pack(side="left", fill="both", expand=True)
        for name in ("pack", "grid", "place", "pack_forget", "grid_forget", "place_forget"):
            setattr(self, name, getattr(self.frame, name))


def log_box(master, height: int = 10) -> ScrollText:
    return ScrollText(master, height=height, font=FONT_MONO, wrap="none", state="disabled")


def append_log(box: tk.Text, lines: list[str], limit: int = 4000) -> None:
    if not lines:
        return
    box.configure(state="normal")
    box.insert("end", "\n".join(lines) + "\n")
    excess = int(box.index("end-1c").split(".")[0]) - limit
    if excess > 0:
        box.delete("1.0", f"{excess}.0")
    box.see("end")
    box.configure(state="disabled")


def set_text(box: tk.Text, text: str) -> None:
    box.configure(state="normal")
    box.delete("1.0", "end")
    box.insert("1.0", text)
    box.configure(state="disabled")


def section(master, title: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(master, text=f" {title} ", padding=8)
