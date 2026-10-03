"""Learn tab: play with one qubit, see what entanglement does to the outputs, read the beginner guide.

The formulas here are exactly the ones the test suite checks against the simulator
(tests/test_smoke.py::test_one_layer_circuit_has_closed_form), so the sliders show what the
final model's circuit really computes, one qubit at a time.
"""
from __future__ import annotations

import math
import os
import re
import tkinter as tk
from tkinter import ttk

from matplotlib.patches import Circle

from common import FIG_DIR, FONT, INK2, ROOT, SERIES, PlotPanel, ScrollText, section, style_axes


class LearnTab(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=8)
        right = ttk.Frame(self, width=560)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)   # keep the guide column readable however wide the plots want to be
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)

        q = section(left, "One qubit: encode an input, rotate, measure")
        q.pack(fill="both", expand=True)
        self.vars = {}
        for name, label, value in (("x", "input angle x (from the CNN)", 1.0), ("theta", "trained angle theta", 0.4),
                                   ("phi", "trained angle phi", 0.8)):
            row = ttk.Frame(q)
            row.pack(fill="x")
            ttk.Label(row, text=label, width=28).pack(side="left")
            var = tk.DoubleVar(value=value)
            ttk.Scale(row, from_=-math.pi, to=math.pi, variable=var, command=lambda _: self.update_qubit()).pack(
                side="left", fill="x", expand=True)
            self.vars[name] = var
        self.formula = ttk.Label(q, text="", font=("Consolas", 10), justify="left")
        self.formula.pack(anchor="w", pady=4)
        self.qplot = PlotPanel(q, figsize=(7, 2.5), toolbar=False)
        self.qplot.pack(fill="both", expand=True)

        e = section(left, "Eight qubits + CNOT ring: outputs become running products")
        e.pack(fill="both", expand=True, pady=(8, 0))
        row = ttk.Frame(e)
        row.pack(fill="x")
        ttk.Label(row, text="every qubit's own <Z> = c", width=28).pack(side="left")
        self.c = tk.DoubleVar(value=0.8)
        ttk.Scale(row, from_=-1, to=1, variable=self.c, command=lambda _: self.update_ring()).pack(side="left", fill="x", expand=True)
        self.eplot = PlotPanel(e, figsize=(7, 2.6), toolbar=False)
        self.eplot.pack(fill="both", expand=True)

        circ = section(right, "Circuit diagrams")
        circ.pack(fill="x")
        for name, label in (("circuit.png", "Final circuit (8 qubits, 1 layer)"),
                            ("circuit_8x3.png", "Default circuit (8 x 3, re-uploading)")):
            ttk.Button(circ, text=label, command=lambda n=name: self.show_image(FIG_DIR / n)).pack(fill="x", pady=1)
        guide = section(right, "Beginner guide (EXPLAINED.md)")
        guide.pack(fill="both", expand=True, pady=(8, 0))
        ttk.Button(guide, text="Open in editor", command=lambda: os.startfile(ROOT / "EXPLAINED.md")).pack(anchor="e")
        self.text = ScrollText(guide, wrap="word", font=FONT, padx=8, pady=8, width=60)
        self.text.pack(fill="both", expand=True)
        self.load_guide()
        self.update_qubit()
        self.update_ring()

    def update_qubit(self) -> None:
        x, th, ph = (self.vars[k].get() for k in ("x", "theta", "phi"))
        # Bloch vector: start |0> = (0, 0, 1); RY(x); RZ(phi); RY(theta)  (the last RZ(omega) changes nothing we measure)
        bx, by, bz = math.sin(x) * math.cos(ph), math.sin(x) * math.sin(ph), math.cos(x)
        X, Z = bx * math.cos(th) + bz * math.sin(th), -bx * math.sin(th) + bz * math.cos(th)
        self.formula.configure(text=f"<Z> = cos(theta) cos(x) - sin(theta) cos(phi) sin(x)\n"
                                    f"    = {math.cos(th):+.2f} x {math.cos(x):+.2f} - {math.sin(th):+.2f} x "
                                    f"{math.cos(ph):+.2f} x {math.sin(x):+.2f} = {Z:+.3f}")
        fig = self.qplot.figure
        fig.clear()
        a1, a2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        a1.add_patch(Circle((0, 0), 1, fill=False, color=INK2))
        a1.annotate("", xy=(X, Z), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=2))
        a1.text(0, 1.08, "|0>   <Z> = +1", ha="center", fontsize=8)
        a1.text(0, -1.2, "|1>   <Z> = -1", ha="center", fontsize=8)
        a1.set_xlim(-1.3, 1.3)
        a1.set_ylim(-1.3, 1.3)
        a1.set_aspect("equal")
        a1.axis("off")
        a1.set_title(f"Bloch vector (x-z slice); y = {by:+.2f}", fontsize=8)
        xs = [-math.pi + 2 * math.pi * i / 200 for i in range(201)]
        style_axes(a2)
        a2.plot(xs, [math.cos(th) * math.cos(t) - math.sin(th) * math.cos(ph) * math.sin(t) for t in xs], color=SERIES[0])
        a2.plot([x], [Z], "o", color=SERIES[1])
        a2.set_ylim(-1.05, 1.05)
        a2.set_xlabel("input x", fontsize=8)
        a2.set_title("<Z> as a function of the input: a shifted cosine", fontsize=8)
        self.qplot.draw()

    def update_ring(self) -> None:
        c = self.c.get()
        outs = [c ** 7] + [c ** (k + 1) for k in range(1, 8)]   # out_0 = c1...c7, out_k = c0...ck
        fig = self.eplot.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        style_axes(ax)
        ax.bar(range(8), [c] * 8, width=0.35, color=SERIES[2], label="without CNOTs: each output = c")
        ax.bar([i + 0.37 for i in range(8)], outs, width=0.35, color=SERIES[0], label="with the CNOT ring: products")
        ax.set_ylim(-1, 1)
        ax.set_xticks([i + 0.18 for i in range(8)], [f"qubit {i}" for i in range(8)], fontsize=7)
        ax.set_ylabel("<Z>", fontsize=8)
        ax.set_title("green: without CNOTs, every output is c     blue: with the CNOT ring, running products of c",
                     fontsize=8)
        self.eplot.draw()

    def load_guide(self) -> None:
        self.text.tag_configure("h1", font=("Segoe UI", 15, "bold"), spacing1=8)
        self.text.tag_configure("h2", font=("Segoe UI", 12, "bold"), spacing1=10)
        self.text.tag_configure("code", font=("Consolas", 9), background="#f2f1ed")
        try:
            md = (ROOT / "EXPLAINED.md").read_text(encoding="utf-8")
        except OSError:
            md = "EXPLAINED.md not found."
        in_code, paragraph = False, []

        def flush() -> None:   # the file wraps its paragraphs at ~110 characters; let the window wrap them instead
            if paragraph:
                self.text.insert("end", " ".join(paragraph) + "\n")
                paragraph.clear()

        for line in md.splitlines():
            if line.startswith("```"):
                flush()
                in_code = not in_code
                continue
            if in_code:
                self.text.insert("end", line + "\n", "code")
                continue
            if line.startswith(("<!--", "![")):
                continue
            clean = re.sub(r"\*\*|`", "", line).strip()
            if line.startswith("#"):
                flush()
                self.text.insert("end", clean.lstrip("# ") + "\n", "h1" if line.startswith("# ") else "h2")
            elif not clean or line.lstrip().startswith(("-", "|")) or re.match(r"\d+\.\s", clean):
                flush()   # blank lines, list items and table rows keep their own line
                self.text.insert("end", clean + "\n")
            else:
                paragraph.append(clean)
        flush()
        self.text.configure(state="disabled")

    def show_image(self, path) -> None:
        if not path.exists():
            return
        from PIL import Image, ImageTk

        top = tk.Toplevel(self)
        top.title(path.name)
        img = Image.open(path)
        img.thumbnail((self.winfo_screenwidth() - 80, self.winfo_screenheight() - 160), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label = ttk.Label(top, image=photo)
        label.image = photo
        label.pack()
