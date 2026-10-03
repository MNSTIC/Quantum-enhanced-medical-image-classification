"""Score saved checkpoints on the test set, then build every results table and figure.

    python src/evaluate.py              # test scores of the final models (+ extensions, if trained)
    python src/evaluate.py --val-only   # Phase 3 sweep table from validation metrics; never opens test
    python src/evaluate.py --noise      # also Phase 5.1: finite shots and depolarising noise (slow;
                                        # saved, and reused by later calls without --noise)

This is the ONLY script that loads the test split. Each checkpoint is scored once: its test
probabilities are cached in test_predictions.csv and reused by later calls, so rerunning this
script only rebuilds tables and figures. Nothing here selects a model, epoch, seed or threshold.

Tables go to runs/results.md and are copied into README.md / REPORT.md between
`<!-- results:KEY -->` and `<!-- /results:KEY -->` markers, so no number is typed by hand.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from medmnist import Evaluator

import figures
from config import DATA_DIR, FIG_DIR, FINAL_HYBRID, MATCHED_HYBRID, ROOT, RUNS_DIR, Config, config_from_dict
from torch.utils.data import DataLoader

from data import load_split, make_test_loader, pos_weight
from engine import bootstrap_ci, compute_metrics, predict, predict_tta, weighted_bce
from models import build_model
from quantum import HardwareNoise, draw_circuit, make_quantum_layer
from sweep import STAGES
from utils import load_json, save_json

# Published PneumoniaMNIST test results: MedMNIST v2 paper (Yang et al., Scientific Data 10, 41, 2023), Table 3.
BASELINES = {
    "ResNet-18 (28x28)": (0.944, 0.854), "ResNet-18 (224x224)": (0.956, 0.864),
    "ResNet-50 (28x28)": (0.948, 0.854), "ResNet-50 (224x224)": (0.962, 0.884),
    "auto-sklearn": (0.942, 0.855), "AutoKeras": (0.947, 0.878), "Google AutoML Vision": (0.991, 0.946),
}
REFERENCE_LINES = ("ResNet-18 (28x28)", "ResNet-50 (224x224)", "Google AutoML Vision")
DOCS = (ROOT / "README.md", ROOT / "REPORT.md", ROOT / "EXPLAINED.md")


# ---------------------------------------------------------------------------------------- helpers
def label(cfg: Config) -> str:
    base = {"hybrid": f"Hybrid {cfg.n_qubits}q x {cfg.n_layers}L", "classical": f"Classical control ({cfg.n_qubits} units)",
            "quanv": "Quanvolution front end", "quanv_classical": "Conv 2x2 control"}[cfg.mode]
    base += "" if cfg.entangle or cfg.mode != "hybrid" else ", no entanglement"
    base += "" if cfg.reupload or cfg.mode != "hybrid" else ", no re-uploading"
    base += "" if cfg.size == 28 else f", {cfg.size}x{cfg.size}"
    base += "" if cfg.epochs == 30 else f", {cfg.epochs} epochs"
    return base + ("" if not cfg.ema else ", EMA")


def inert_angles(cfg: Config) -> int:
    """Last-layer RZ(omega) circuit angles that provably cannot change any <Z> (see tests/test_smoke.py)."""
    return cfg.n_qubits if cfg.mode == "hybrid" else 4 if cfg.mode == "quanv" else 0


def pm(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.iloc[0]:.3f} (1 seed)"


def ci(c: list[float]) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def seed_dirs(cfg: Config) -> list[Path]:
    gdir = RUNS_DIR / cfg.run_name()
    return sorted((d for d in gdir.glob("seed*") if (d / "metrics.json").exists()), key=lambda d: int(d.name[4:]))


def inject(key: str, text: str) -> None:
    """Replace the block between the results markers in README.md, REPORT.md and EXPLAINED.md."""
    start, end = f"<!-- results:{key} -->", f"<!-- /results:{key} -->"
    for path in DOCS:
        if not path.exists():
            continue
        s = path.read_text(encoding="utf-8")
        if start in s and end in s:
            a, b = s.index(start) + len(start), s.index(end)
            path.write_text(s[:a] + "\n" + text.strip() + "\n" + s[b:], encoding="utf-8")


# ------------------------------------------------------------------------------- test-set scoring
@cache
def test_loader(size: int):
    return make_test_loader(size)        # asserts the test class counts (624 = 234 normal + 390 pneumonia)


@cache
def official(size: int) -> Evaluator:
    test_loader(size)   # downloads this resolution's file if missing; Evaluator only reads from disk
    return Evaluator("pneumoniamnist", "test", size=None if size == 28 else size, root=str(DATA_DIR))


def load_model(run_dir: Path) -> tuple[Config, torch.nn.Module]:
    """Rebuild the architecture from config.json and load the trained weights (state_dict)."""
    cfg = config_from_dict(load_json(run_dir / "config.json"))
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    return cfg, model.eval()


def test_probs(run_dir: Path, size: int, y: np.ndarray, tta: bool = False) -> np.ndarray:
    """P(pneumonia) for the 624 test images, in dataset order; computed once per checkpoint.

    tta=True averages over the 7 test-time views (Phase 7) and is cached in its own file.
    """
    f, ckpt = run_dir / ("test_predictions_tta.csv" if tta else "test_predictions.csv"), run_dir / "best.pt"
    # Reuse the cache unless the checkpoint was retrained after it. Checkpoints are not in git, so a fresh
    # clone rebuilds every table from the committed predictions.
    if f.exists() and (not ckpt.exists() or f.stat().st_mtime > ckpt.stat().st_mtime):
        df = pd.read_csv(f)
    else:
        _, model = load_model(run_dir)
        if tta:
            df = pd.DataFrame({"label": y, "prob": predict_tta(model, test_loader(size))})
        else:
            p, y_loader, _ = predict(model, test_loader(size))
            df = pd.DataFrame({"label": y_loader, "prob": p})
        df.to_csv(f, index_label="index")
    assert np.array_equal(df["label"].to_numpy(), y), "test labels out of dataset order"
    return df["prob"].to_numpy()


def score_group(cfg: Config, tta: bool = False) -> dict | None:
    """Per-seed test metrics, the seed-ensemble (mean probability) and its bootstrap CIs."""
    dirs = seed_dirs(cfg)
    if not dirs:
        print(f"  {cfg.run_name()}: no finished runs, skipped")
        return None
    ev = official(cfg.size)
    y = ev.labels.reshape(-1).astype(int)
    rows, probs = [], []
    for d in dirs:
        p = test_probs(d, cfg.size, y, tta)
        m = compute_metrics(y, p)
        auc, acc = ev.evaluate(p)        # the official MedMNIST numbers must equal ours
        assert abs(auc - m["auc"]) < 1e-9 and abs(acc - m["acc"]) < 1e-9, (d, auc, acc, m)
        save_json(m, d / ("test_metrics_tta.json" if tta else "test_metrics.json"))
        rows.append({"seed": int(d.name[4:]), **m, "val_auc": load_json(d / "metrics.json")["val_auc"]})
        probs.append(p)
    seeds = pd.DataFrame(rows)
    ens_p = np.mean(probs, axis=0)
    info = load_json(dirs[0] / "metrics.json")
    r = {"cfg": cfg, "name": cfg.run_name() + ("_tta" if tta else ""), "label": label(cfg) + (" + TTA" if tta else ""),
         "y": y, "seeds": seeds, "ens_p": ens_p,
         "ensemble": {**compute_metrics(y, ens_p), **bootstrap_ci(y, ens_p)},
         "n_params": info["n_params"], "n_middle": info["n_params_probe"],
         "histories": [pd.read_csv(d / "history.csv") for d in dirs],
         "epoch_s": float(np.mean([load_json(d / "metrics.json")["epoch_s_mean"] for d in dirs])),
         "train_min": float(np.mean([load_json(d / "metrics.json")["train_s"] for d in dirs])) / 60,
         "fastest_min": float(np.min([load_json(d / "metrics.json")["train_s"] for d in dirs])) / 60}
    # Section 10 of the brief: implausibly good numbers mean leakage or a metric bug until shown otherwise.
    if seeds["acc"].mean() > 0.96 or seeds["auc"].mean() > seeds["val_auc"].mean():
        print(f"  WARNING {r['name']}: test ACC > 0.96 or test AUC above validation AUC; check for leakage")
    print(f"  {r['name']}: {len(dirs)} seeds, test AUC {pm(seeds['auc'])}, ACC {pm(seeds['acc'])}")
    return r


# ----------------------------------------------------------------------------------------- tables
def main_table(res: list[dict]) -> str:
    out = ["| Model | Params: total / compared block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | "
           "Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        s, e = r["seeds"], r["ensemble"]
        eff = r["n_middle"] - inert_angles(r["cfg"])
        out.append(f"| {r['label']} | {r['n_params']:,} / {r['n_middle']} ({eff}) | {len(s)} | {pm(s['auc'])} | "
                   f"{pm(s['acc'])} | {pm(s['sensitivity'])} | {pm(s['specificity'])} | {pm(s['balanced_acc'])} | "
                   f"{e['auc']:.3f} {ci(e['auc_ci'])} | {e['acc']:.3f} {ci(e['acc_ci'])} |")
    out.append("\nMean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. "
               "Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. "
               "Compared block = the 8 -> 8 middle block, or the 2x2 front end for the quanvolution models. "
               "'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.")
    return "\n".join(out)


def seed_table(res: list[dict]) -> str:
    seeds = sorted({k for r in res for k in r["seeds"]["seed"]})
    out = ["| Model | Metric | " + " | ".join(f"seed {k}" for k in seeds) + " |",
           "|---|---|" + "---|" * len(seeds)]
    for r in res:
        by_seed = r["seeds"].set_index("seed")
        for m in ("val_auc", "auc", "acc", "sensitivity", "specificity"):
            name = {"val_auc": "val AUC", "auc": "test AUC", "acc": "test ACC"}.get(m, f"test {m}")
            vals = [f"{by_seed.loc[k, m]:.3f}" if k in by_seed.index else "" for k in seeds]
            out.append(f"| {r['label']} | {name} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def compare(a: dict, b: dict) -> str:
    """Rule fixed in the brief: a mean difference smaller than the seed std, or overlapping
    ensemble CIs, is reported as 'no significant difference'. The larger of the two seed stds is
    used (the conservative choice)."""
    out = [f"**A = {a['label']}, B = {b['label']}**\n",
           "| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |",
           "|---|---|---|---|---|---|---|"]
    paired = a["seeds"].merge(b["seeds"], on="seed", suffixes=("_a", "_b"))
    for m in ("auc", "acc"):
        sa, sb = a["seeds"][m], b["seeds"][m]
        diff, sd = sa.mean() - sb.mean(), np.nanmax([sa.std(ddof=1), sb.std(ddof=1)])
        ca, cb = a["ensemble"][f"{m}_ci"], b["ensemble"][f"{m}_ci"]
        overlap = ca[0] <= cb[1] and cb[0] <= ca[1]
        wins = int((paired[f"{m}_a"] > paired[f"{m}_b"]).sum())
        if abs(diff) < sd or overlap or np.isnan(sd):
            verdict = "no significant difference"
        else:
            verdict = f"{'A' if diff > 0 else 'B'} better"
        out.append(f"| {m.upper()} | {diff:+.4f} | {sd:.4f} | {ci(ca)} vs {ci(cb)} | {'yes' if overlap else 'no'} | "
                   f"{wins} of {len(paired)} | **{verdict}** |")
    return "\n".join(out)


def target_line(r: dict) -> str:
    auc, acc = r["seeds"]["auc"].mean(), r["seeds"]["acc"].mean()
    t_auc, t_acc = BASELINES["ResNet-18 (28x28)"]
    met = "met" if auc > t_auc and acc > t_acc else "NOT met"
    return (f"Target (beat ResNet-18 at 28x28, AUC {t_auc} / ACC {t_acc}): {r['label']} mean test AUC "
            f"{auc:.3f} ({auc - t_auc:+.3f}), ACC {acc:.3f} ({acc - t_acc:+.3f}) over {len(r['seeds'])} seeds: **{met}**.")


def baseline_table() -> str:
    rows = ["| Published method (MedMNIST v2, Table 3) | Test AUC | Test ACC |", "|---|---|---|"]
    rows += [f"| {k} | {auc:.3f} | {acc:.3f} |" for k, (auc, acc) in BASELINES.items()]
    return "\n".join(rows)


def timing_table(res: list[dict]) -> str:
    rows = ["| Model | Seeds | Mean epoch time (s) | Mean training run (min, 30 epochs) | Fastest run (min) |",
            "|---|---|---|---|---|"]
    rows += [f"| {r['label']} | {len(r['seeds'])} | {r['epoch_s']:.1f} | {r['train_min']:.1f} | {r['fastest_min']:.1f} |"
             for r in res]
    rows.append("\nCPU only, 2 PyTorch threads per run; up to 8 runs shared the 20-core CPU at once, "
                "so single-run times on an idle machine are shorter.")
    return "\n".join(rows)


def device_table(b: pd.DataFrame) -> str:
    """CPU vs GPU seconds per epoch, written by src/benchmark.py (single runs on an otherwise idle machine)."""
    rows = ["| Model | Device | CPU threads | PyTorch build | Seconds per epoch |", "|---|---|---|---|---|"]
    for _, r in b.iterrows():
        rows.append(f"| {r['model']} | {r['device']}{' (' + r['gpu'] + ')' if isinstance(r['gpu'], str) and r['gpu'] else ''} | "
                    f"{r['threads']} | {r['torch']} | {r['epoch_s']:.2f} |")
    rows.append("\nOne training epoch plus one validation pass, mean of epochs 2-3 (epoch 1 is warm-up), one run at a "
                "time. Every reported result was trained on the CPU with 2 threads, which is bit-for-bit reproducible; "
                "GPU arithmetic differs slightly.")
    return "\n".join(rows)


def summary_rows(r: dict) -> list[dict]:
    s, e = r["seeds"], r["ensemble"]
    cols = [c for c in s.columns if c != "seed"]
    base = {"group": r["name"], "n_params": r["n_params"], "n_middle": r["n_middle"]}
    rows = [{**base, "row": f"seed{int(k)}", **row.drop("seed").to_dict()} for k, row in s.set_index("seed", drop=False).iterrows()]
    rows.append({**base, "row": "mean", **s[cols].mean().to_dict()})
    rows.append({**base, "row": "std", **s[cols].std(ddof=1).to_dict()})
    rows.append({**base, "row": "ensemble", **{k: v for k, v in e.items() if not k.endswith("_ci")},
                 "auc_ci_lo": e["auc_ci"][0], "auc_ci_hi": e["auc_ci"][1],
                 "acc_ci_lo": e["acc_ci"][0], "acc_ci_hi": e["acc_ci"][1]})
    return rows


def gradient_table() -> str:
    """Barren-plateau check: gradient norm of the circuit angles over training, for every hybrid run."""
    rows = ["| Run group | Runs | Smallest epoch mean | Largest epoch mean | Last epoch (mean over runs) |",
            "|---|---|---|---|---|"]
    for gdir in sorted(RUNS_DIR.glob("hybrid_*")):
        hs = [pd.read_csv(f)["probe_grad_norm"] for f in sorted(gdir.glob("seed*/history.csv"))]
        if hs:
            rows.append(f"| {gdir.name} | {len(hs)} | {min(h.min() for h in hs):.1e} | {max(h.max() for h in hs):.1e} | "
                        f"{np.mean([h.iloc[-1] for h in hs]):.1e} |")
    rows.append("\nL2 norm of the circuit-angle gradients, averaged over the steps of each epoch (before clipping). "
                "A barren plateau would show as values collapsing towards zero.")
    return "\n".join(rows)


# --------------------------------------------------------------------------- validation-only sweep
def sweep_table() -> pd.DataFrame | None:
    """Phase 3 table, built from validation metrics only (metrics.json); never opens the test set."""
    settings, seeds = STAGES["sweep"]
    rows = []
    for cfg in settings:
        ms = [load_json(p) for k in seeds if (p := replace(cfg, seed=k).run_dir() / "metrics.json").exists()]
        if not ms:
            continue
        setting = f"{cfg.n_qubits}q x {cfg.n_layers}L" + ("" if cfg.entangle else ", no entanglement") \
            + ("" if cfg.reupload else ", no re-uploading")
        df = pd.DataFrame(ms)
        rows.append({"setting": setting, "name": cfg.run_name(), "seeds": len(ms), "circuit_params": ms[0]["n_params_probe"],
                     "val_auc_mean": df["val_auc"].mean(), "val_auc_std": df["val_auc"].std(ddof=1),
                     "val_loss_mean": df["val_loss"].mean(), "val_loss_std": df["val_loss"].std(ddof=1),
                     "val_acc_mean": df["val_acc"].mean(), "best_epoch_mean": df["best_epoch"].mean(),
                     "epoch_s": df["epoch_s_mean"].mean()})
    return pd.DataFrame(rows) if rows else None


def sweep_markdown(t: pd.DataFrame) -> str:
    out = ["| Setting | Seeds | Circuit params | Val AUC | Val loss | Val ACC | Mean best epoch | Epoch time (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        out.append(f"| {r['setting']} | {r['seeds']} | {r['circuit_params']} | {r['val_auc_mean']:.4f} ± {r['val_auc_std']:.4f} | "
                   f"{r['val_loss_mean']:.4f} ± {r['val_loss_std']:.4f} | {r['val_acc_mean']:.3f} | "
                   f"{r['best_epoch_mean']:.1f} | {r['epoch_s']:.1f} |")
    out.append("\nValidation split only (524 images); mean ± sample std over seeds 0-2. Checkpoint = epoch with the "
               "best validation AUC. Epoch times measured with up to 8 runs sharing the CPU.")
    return "\n".join(out)


# --------------------------------------------------------------------- Phase 5.1 / 7: noise models
def readout_errors(mean: float, n: int) -> tuple[float, ...]:
    """Unequal per-qubit readout-flip probabilities with the given mean (one fixed random draw).

    Real qubits differ, and only unequal errors can change a ranking: an equal error p on every qubit
    just rescales every output by (1 - 2p), which the linear head turns into a monotone change of the logit.
    """
    return tuple(float(v) for v in np.random.default_rng(1234).uniform(0, 2 * mean, n))


def device_noise(scale: float, n: int) -> HardwareNoise:
    """Order-of-magnitude device profile x `scale`: 0.1% single-qubit and 1% two-qubit gate error,
    0.1% damping per layer, 2% mean readout error (roughly today's superconducting chips at scale 1)."""
    return HardwareNoise(p1=1e-3 * scale, p2=1e-2 * scale, gamma=1e-3 * scale, readout=readout_errors(0.02 * scale, n))


# setting name -> (kind of error, options understood by noisy_layer)
NOISE_SETTINGS: dict[str, tuple[str, dict]] = {
    "shots=1000": ("sampling", {"shots": 1000}),
    "p=0.001": ("depolarising after each layer", {"noise_p": 0.001}),
    "p=0.01": ("depolarising after each layer", {"noise_p": 0.01}),
    "p=0.05": ("depolarising after each layer", {"noise_p": 0.05}),
    "readout 1%": ("unequal readout errors", {"readout": 0.01}),
    "readout 3%": ("unequal readout errors", {"readout": 0.03}),
    "readout 5%": ("unequal readout errors", {"readout": 0.05}),
    "device x1": ("gate + damping + readout errors", {"device_scale": 1}),
    "device x3": ("gate + damping + readout errors", {"device_scale": 3}),
    "device x10": ("gate + damping + readout errors", {"device_scale": 10}),
    "miscalibration 0.02 rad": ("coherent angle errors", {"sigma": 0.02}),
    "miscalibration 0.05 rad": ("coherent angle errors", {"sigma": 0.05}),
    "miscalibration 0.1 rad": ("coherent angle errors", {"sigma": 0.1}),
}


def noisy_layer(cfg: Config, trained: torch.nn.Module, setting: str, seed: int) -> torch.nn.Module:
    """A copy of the trained circuit layer running under one noise setting (evaluation only).

    Shots and density-matrix noise need a QNode with diff_method=None; nothing here is trained.
    """
    opts = NOISE_SETTINGS[setting][1]
    base = dict(entangle=cfg.entangle, reupload=cfg.reupload, diff_method=None)
    if "sigma" in opts:
        # Coherent error: every gate angle is off by a fixed random amount, as on a miscalibrated
        # device. Exact simulator; the trained angles themselves are perturbed (fixed draw per seed).
        layer = make_quantum_layer(cfg.n_qubits, cfg.n_layers, **base)
        w = trained.state_dict()["weights"]
        g = torch.Generator().manual_seed(seed)
        layer.load_state_dict({"weights": w + opts["sigma"] * torch.randn(w.shape, generator=g)})
        return layer
    if "readout" in opts:
        kw = {"device": "default.mixed", "hw": HardwareNoise(readout=readout_errors(opts["readout"], cfg.n_qubits))}
    elif "device_scale" in opts:
        kw = {"device": "default.mixed", "hw": device_noise(opts["device_scale"], cfg.n_qubits)}
    elif "noise_p" in opts:
        kw = {"device": "default.mixed", "noise_p": opts["noise_p"]}
    else:
        kw = {"shots": opts["shots"], "seed": seed}   # finite shots: the sampling is seeded for reproducibility
    layer = make_quantum_layer(cfg.n_qubits, cfg.n_layers, **base, **kw)
    layer.load_state_dict(trained.state_dict())   # same trained angles, different simulator
    return layer


@cache
def noise_loader(size: int) -> DataLoader:
    """The test images in batches of 32 for the density-matrix simulator.

    Each image's 8-qubit density matrix is 256 x 256 complex numbers and the simulator copies it several times
    per gate, so 256-image batches need gigabytes per process. The simulation is deterministic and per image,
    so the batch size does not change the results. (Finite-shot settings keep the standard loader: their
    random draws are tied to the batches.)
    """
    return DataLoader(test_loader(size).dataset, batch_size=32, shuffle=False, num_workers=0)


def noise_for_seed(d: Path, threads: int = 2) -> pd.DataFrame:
    """Test probabilities of one trained hybrid under every noise setting; cached in noise_predictions.csv.

    Runs in its own process (see noise_eval), so it sets its own thread count.
    """
    torch.set_num_threads(threads)
    cfg = config_from_dict(load_json(d / "config.json"))
    y = official(cfg.size).labels.reshape(-1).astype(int)
    f = d / "noise_predictions.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame({"label": y, "exact": test_probs(d, cfg.size, y)})
    for setting in NOISE_SETTINGS:
        if setting in df:
            continue
        _, model = load_model(d)
        model.middle = noisy_layer(cfg, model.middle, setting, int(d.name[4:]))
        mixed = bool(NOISE_SETTINGS[setting][1].keys() & {"noise_p", "readout", "device_scale"})
        df[setting] = predict(model, noise_loader(cfg.size) if mixed else test_loader(cfg.size))[0]
        df.to_csv(f, index=False)   # saved after every setting, so an interrupted run resumes here
        print(f"  noise {d.parent.name}/{d.name} {setting}: AUC {compute_metrics(y, df[setting].to_numpy())['auc']:.4f}",
              flush=True)
    return df


def noise_eval(cfg: Config, workers: int) -> pd.DataFrame | None:
    """Paired effect of every noise setting on test AUC/ACC, over all seeds of one trained hybrid.

    Seeds are independent, so they run in parallel processes. Every seed's results are cached and
    seeded, so the numbers do not depend on how many processes are used.
    """
    from concurrent.futures import ProcessPoolExecutor

    dirs = seed_dirs(cfg)
    if not dirs:
        return None
    with ProcessPoolExecutor(min(workers, len(dirs))) as pool:
        per_seed = list(pool.map(noise_for_seed, dirs))
    y = per_seed[0]["label"].to_numpy()
    base = pd.DataFrame([compute_metrics(y, df["exact"].to_numpy()) for df in per_seed])
    rows = []
    for setting in ["exact", *NOISE_SETTINGS]:
        m = pd.DataFrame([compute_metrics(y, df[setting].to_numpy()) for df in per_seed])
        d_auc, d_acc = m["auc"] - base["auc"], m["acc"] - base["acc"]   # paired: same seed, noisy minus exact
        kind, opts = NOISE_SETTINGS.get(setting, ("none", {}))
        rows.append({"setting": "exact simulation" if setting == "exact" else setting, "category": kind,
                     "p": opts.get("noise_p", 0.0),
                     "auc_mean": m["auc"].mean(), "auc_std": m["auc"].std(ddof=1),
                     "acc_mean": m["acc"].mean(), "acc_std": m["acc"].std(ddof=1),
                     "d_auc_mean": d_auc.mean(), "d_auc_std": d_auc.std(ddof=1),
                     "d_acc_mean": d_acc.mean(), "d_acc_std": d_acc.std(ddof=1)})
    return pd.DataFrame(rows)


def noise_markdown(t: pd.DataFrame, n: int) -> str:
    out = ["| Circuit simulation | Kind of error | Test AUC | Test ACC | Δ AUC vs exact (paired) | Δ ACC vs exact (paired) |",
           "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        out.append(f"| {r['setting']} | {r['category']} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
                   f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f} | {r['d_auc_mean']:+.4f} ± {r['d_auc_std']:.4f} | "
                   f"{r['d_acc_mean']:+.4f} ± {r['d_acc_std']:.4f} |")
    out.append(f"\nSame trained weights, no retraining; mean ± std over {n} seeds; Δ = noisy minus exact for the same "
               "seed. shots: each <Z> estimated from 1,000 measurements. p: depolarising channel on every qubit after "
               "every layer. readout: unequal per-qubit bit-flip errors with this mean. device xk: 0.1% single-qubit "
               "and 1% two-qubit gate errors, 0.1% amplitude and phase damping per layer and 2% mean readout error, "
               "all multiplied by k. miscalibration: every circuit angle off by a fixed random N(0, sigma) amount. "
               "Order-of-magnitude illustrations, not calibrated models of a real device.")
    return "\n".join(out)


# ---------------------------------------------------------- secondary: validation-tuned threshold
def tuned_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1 on validation data.

    Validation is nearly separable, so several thresholds can tie; the median of the tied ones is used.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(np.median(thr[np.isclose(j, j.max()) & np.isfinite(thr)]))


def threshold_table(res: list[dict]) -> str:
    """Secondary result: each seed's threshold is tuned on its own validation predictions, then applied
    to the test set, next to the fixed 0.5 threshold, which stays the headline."""
    out = ["| Model | Val-tuned threshold | Test ACC at 0.5 | Test ACC at tuned | Sensitivity at tuned | "
           "Specificity at tuned | Balanced acc. at tuned | Ensemble ACC at tuned |",
           "|---|---|---|---|---|---|---|---|"]
    for r in res:
        rows, val_probs = [], []
        for d in seed_dirs(r["cfg"]):
            v = pd.read_csv(d / "val_predictions.csv")
            y_val = v["label"].to_numpy()
            val_probs.append(v["prob"].to_numpy())
            t = tuned_threshold(y_val, val_probs[-1])
            rows.append({"t": t, **compute_metrics(r["y"], pd.read_csv(d / "test_predictions.csv")["prob"].to_numpy(), t)})
        df = pd.DataFrame(rows)
        t_ens = tuned_threshold(y_val, np.mean(val_probs, axis=0))
        ens = compute_metrics(r["y"], r["ens_p"], threshold=t_ens)
        out.append(f"| {r['label']} | {pm(df['t'])} | {pm(r['seeds']['acc'])} | {pm(df['acc'])} | {pm(df['sensitivity'])} | "
                   f"{pm(df['specificity'])} | {pm(df['balanced_acc'])} | {ens['acc']:.3f} (t = {t_ens:.3f}) |")
    out.append("\nSecondary result. Each threshold is chosen on validation data only (largest sensitivity + "
               "specificity); the test set never influences it. The headline numbers keep the fixed 0.5 threshold.")
    return "\n".join(out)


# ------------------------------------------------------------- Phase 7: post-hoc improvement study
PHASE7_V1 = (FINAL_HYBRID, Config(mode="classical", n_qubits=FINAL_HYBRID.n_qubits))   # the main recipe
PHASE7_V2 = tuple(STAGES["phase7"][0])                                                # 60 epochs + EMA
VARIANTS = (("v1", 0, False), ("v1 + TTA", 0, True), ("v2", 1, False), ("v2 + TTA", 1, True))


@cache
def val_loader() -> DataLoader:
    return DataLoader(load_split("val"), batch_size=256, shuffle=False, num_workers=0)


def val_probs(d: Path, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    """(labels, P(pneumonia)) on the validation set for one checkpoint; the TTA version is cached."""
    v = pd.read_csv(d / "val_predictions.csv")
    if not tta:
        return v["label"].to_numpy(), v["prob"].to_numpy()
    f, ckpt = d / "val_predictions_tta.csv", d / "best.pt"
    if f.exists() and (not ckpt.exists() or f.stat().st_mtime > ckpt.stat().st_mtime):
        return v["label"].to_numpy(), pd.read_csv(f)["prob"].to_numpy()
    _, model = load_model(d)
    p = predict_tta(model, val_loader())
    pd.DataFrame({"label": v["label"], "prob": p}).to_csv(f, index_label="index")
    return v["label"].to_numpy(), p


def phase7_val() -> pd.DataFrame | None:
    """Validation scores of every Phase 7 variant, per seed. Decides Phase 7; never opens the test set."""
    from sklearn.metrics import roc_auc_score

    pw = pos_weight(load_split("train").labels)
    rows = []
    for model, cfgs in zip(("hybrid", "classical"), zip(PHASE7_V1, PHASE7_V2)):
        for variant, k, tta in VARIANTS:
            for d in seed_dirs(cfgs[k]):
                y, p = val_probs(d, tta)
                rows.append({"model": model, "variant": variant, "seed": int(d.name[4:]),
                             "val_auc": roc_auc_score(y, p), "val_loss": weighted_bce(y, p, pw)})
    return pd.DataFrame(rows) if rows else None


def phase7_val_markdown(t: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    """Validation table plus the verdict of the rule fixed in DECISIONS.md before these numbers existed."""
    out = ["| Model | Variant | Seeds | Val AUC | Val loss | Δ val AUC vs v1 | Δ val loss vs v1 |", "|---|---|---|---|---|---|---|"]
    better: dict[str, list[bool]] = {"TTA": [], "recipe v2": []}
    for model in ("hybrid", "classical"):
        base = t[(t["model"] == model) & (t["variant"] == "v1")]
        for variant, _, _ in VARIANTS:
            s = t[(t["model"] == model) & (t["variant"] == variant)]
            if s.empty:
                continue
            d_auc, d_loss = s["val_auc"].mean() - base["val_auc"].mean(), s["val_loss"].mean() - base["val_loss"].mean()
            out.append(f"| {model} | {variant} | {len(s)} | {s['val_auc'].mean():.4f} ± {s['val_auc'].std(ddof=1):.4f} | "
                       f"{s['val_loss'].mean():.4f} ± {s['val_loss'].std(ddof=1):.4f} | {d_auc:+.4f} | {d_loss:+.4f} |")
            if variant in ("v1 + TTA", "v2"):
                better["TTA" if variant == "v1 + TTA" else "recipe v2"].append(d_loss < 0 and d_auc >= 0)
    adopted = {k: len(v) == 2 and all(v) for k, v in better.items()}
    out.append("\nValidation only. v1 = the main recipe (30 epochs); v2 = 60 epochs + weight averaging (EMA); TTA = "
               "probability averaged over 7 shifted/rotated views. Rule fixed in advance: a candidate is adopted only "
               "if, for both the hybrid and the control, it lowers mean validation loss without lowering mean "
               "validation AUC.\n")
    out += [f"- **{k}: {'adopted' if v else 'not adopted'}** by that rule." for k, v in adopted.items()]
    return "\n".join(out), adopted


def phase7_test(adopted: dict[str, bool]) -> tuple[str, list[dict]]:
    """Post-hoc test scores of every Phase 7 variant, and hybrid vs control under the validated recipe."""
    res = {}
    for model, cfgs in zip(("hybrid", "classical"), zip(PHASE7_V1, PHASE7_V2)):
        for variant, k, tta in VARIANTS:
            r = score_group(cfgs[k], tta)
            if r:
                res[model, variant] = r
    best = ("v2" if adopted.get("recipe v2") else "v1") + (" + TTA" if adopted.get("TTA") else "")
    text = ("**Post-hoc, exploratory**: the test set had already been read once (Phase 4). These rows are reported "
            "whatever the validation verdict, and none of them replaces the headline result.\n\n"
            + main_table(list(res.values())))
    if ("hybrid", best) in res and ("classical", best) in res:
        text += f"\n\nHybrid vs classical control with the variant chosen on validation data ({best}):\n\n" \
                + compare(res["hybrid", best], res["classical", best])
    for variant in ("v2", "v1 + TTA"):   # every other variant, for completeness (not a selection)
        if variant != best and ("hybrid", variant) in res and ("classical", variant) in res:
            text += f"\n\nThe same comparison for {variant}:\n\n" + compare(res["hybrid", variant], res["classical", variant])
    return text, list(res.values())


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-only", action="store_true", help="only the validation sweep table; never load test")
    ap.add_argument("--noise", action="store_true", help="also run the Phase 5.1 shots/noise evaluation")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads for this process")
    ap.add_argument("--workers", type=int, default=5, help="parallel processes for --noise (one seed each, 2 threads)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(args.threads)
    FIG_DIR.mkdir(exist_ok=True)
    sections: dict[str, str] = {}

    for cfg, stem in ((FINAL_HYBRID, "circuit"), (MATCHED_HYBRID, "circuit_8x3")):
        (FIG_DIR / f"{stem}.txt").write_text(draw_circuit(cfg.n_qubits, cfg.n_layers, FIG_DIR / f"{stem}.png"),
                                             encoding="utf-8")
    sweep = sweep_table()
    if sweep is not None:
        sweep.to_csv(RUNS_DIR / "sweep_summary.csv", index=False)
        figures.sweep_bars(sweep, FIG_DIR / "sweep_val_auc.png")
        sections["sweep"] = sweep_markdown(sweep)
    sections["gradients"] = gradient_table()   # training histories only; no test data
    p7 = phase7_val()                            # validation only: decides Phase 7 before any Phase 7 test score
    adopted = {}
    if p7 is not None:
        p7.to_csv(RUNS_DIR / "phase7_val.csv", index=False)
        sections["phase7_val"], adopted = phase7_val_markdown(p7)

    if not args.val_only:
        print("scoring on the test set:")
        main_res = [r for r in map(score_group, STAGES["final"][0]) if r]
        matched = [r for r in map(score_group, STAGES["matched"][0]) if r]
        ext = {k: [r for r in map(score_group, STAGES[k][0]) if r] for k in ("quanv", "s64")}
        everything = main_res + matched + ext["quanv"] + ext["s64"]
        if matched and len(main_res) > 1:   # secondary, declared before test scoring: 72 vs 72 parameters
            sections["matched"] = main_table(matched + main_res[1:2]) + "\n\n" + compare(matched[0], main_res[1])
        if everything:
            pd.DataFrame([row for r in everything for row in summary_rows(r)]).to_csv(RUNS_DIR / "summary.csv", index=False)
            sections["timing"] = timing_table(everything)
        if main_res:
            sections["main"] = main_table(main_res)
            sections["seeds"] = seed_table(main_res)
            sections["target"] = target_line(main_res[0])
            sections["baselines"] = baseline_table()
            if len(main_res) == 3:
                sections["compare"] = compare(main_res[0], main_res[1]) + "\n\n" + compare(main_res[0], main_res[2])
            figures.training_curves({r["label"]: r["histories"] for r in main_res}, FIG_DIR / "training_curves.png")
            figures.roc_curves({r["label"]: (r["y"], r["ens_p"], r["ensemble"]["auc"]) for r in main_res[:2]},
                               FIG_DIR / "roc_test.png")
            figures.confusion_matrices({r["label"]: r["ensemble"] for r in main_res}, FIG_DIR / "confusion_matrices.png")
            figures.score_bars({r["label"]: {"auc": r["seeds"]["auc"], "acc": r["seeds"]["acc"]} for r in main_res},
                               {k: BASELINES[k] for k in REFERENCE_LINES}, FIG_DIR / "test_scores.png")
        for key, res in ext.items():
            if res:
                sections[key] = main_table(res) + ("\n\n" + compare(res[0], res[1]) if len(res) == 2 else "")
        # The 1-layer final circuit has noise only before its measurement, which rescales every <Z> by the
        # same factor; the 3-layer circuit also has noise between layers, the more informative case.
        # Without --noise, the summaries saved by an earlier --noise run are reused (the simulation is slow).
        for cfg, key in ((FINAL_HYBRID, "noise"), (MATCHED_HYBRID, "noise_matched")):
            saved = RUNS_DIR / f"{key}_summary.csv"
            noise = noise_eval(cfg, args.workers) if args.noise else pd.read_csv(saved) if saved.exists() else None
            if noise is not None and "d_auc_mean" in noise:   # older summaries lack the paired columns
                if args.noise:
                    noise.to_csv(saved, index=False)   # only when recomputed; re-saving would just reformat floats
                figures.noise_effects(noise, FIG_DIR / f"{key}.png", label(cfg))
                sections[key] = noise_markdown(noise, len(seed_dirs(cfg)))
        if main_res:
            sections["threshold"] = threshold_table(main_res + matched)
        if p7 is not None:
            sections["phase7"], p7_res = phase7_test(adopted)
            known = {r["name"] for r in everything}
            extra = [r for r in p7_res if r["name"] not in known]   # the two v1 groups are already in the summary
            pd.DataFrame([row for r in everything + extra for row in summary_rows(r)]).to_csv(
                RUNS_DIR / "summary.csv", index=False)
    bench = RUNS_DIR / "device_benchmark.csv"
    if bench.exists():
        sections["devices"] = device_table(pd.read_csv(bench))

    md = "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
    if not args.val_only or not (RUNS_DIR / "results.md").exists():
        (RUNS_DIR / "results.md").write_text("# Generated results (do not edit; rebuilt by src/evaluate.py)\n\n" + md,
                                             encoding="utf-8")
    for k, v in sections.items():
        inject(k, v)
    print(md)


if __name__ == "__main__":
    main()
