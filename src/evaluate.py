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
from data import make_test_loader
from engine import bootstrap_ci, compute_metrics, predict
from models import build_model
from quantum import draw_circuit, make_quantum_layer
from sweep import STAGES
from utils import load_json, save_json

# Published PneumoniaMNIST test results: MedMNIST v2 paper (Yang et al., Scientific Data 10, 41, 2023), Table 3.
BASELINES = {
    "ResNet-18 (28x28)": (0.944, 0.854), "ResNet-18 (224x224)": (0.956, 0.864),
    "ResNet-50 (28x28)": (0.948, 0.854), "ResNet-50 (224x224)": (0.962, 0.884),
    "auto-sklearn": (0.942, 0.855), "AutoKeras": (0.947, 0.878), "Google AutoML Vision": (0.991, 0.946),
}
REFERENCE_LINES = ("ResNet-18 (28x28)", "ResNet-50 (224x224)", "Google AutoML Vision")
NOISE_SETTINGS = {"shots=1000": {"shots": 1000}, "p=0.001": {"noise_p": 0.001},
                  "p=0.01": {"noise_p": 0.01}, "p=0.05": {"noise_p": 0.05}}
DOCS = (ROOT / "README.md", ROOT / "REPORT.md", ROOT / "EXPLAINED.md")


# ---------------------------------------------------------------------------------------- helpers
def label(cfg: Config) -> str:
    base = {"hybrid": f"Hybrid {cfg.n_qubits}q x {cfg.n_layers}L", "classical": f"Classical control ({cfg.n_qubits} units)",
            "quanv": "Quanvolution front end", "quanv_classical": "Conv 2x2 control"}[cfg.mode]
    base += "" if cfg.entangle or cfg.mode != "hybrid" else ", no entanglement"
    base += "" if cfg.reupload or cfg.mode != "hybrid" else ", no re-uploading"
    return base + ("" if cfg.size == 28 else f", {cfg.size}x{cfg.size}")


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


def test_probs(run_dir: Path, size: int, y: np.ndarray) -> np.ndarray:
    """P(pneumonia) for the 624 test images, in dataset order; computed once per checkpoint."""
    f, ckpt = run_dir / "test_predictions.csv", run_dir / "best.pt"
    # Reuse the cache unless the checkpoint was retrained after it. Checkpoints are not in git, so a fresh
    # clone rebuilds every table from the committed predictions.
    if f.exists() and (not ckpt.exists() or f.stat().st_mtime > ckpt.stat().st_mtime):
        df = pd.read_csv(f)
    else:
        _, model = load_model(run_dir)
        p, y_loader, _ = predict(model, test_loader(size))
        df = pd.DataFrame({"label": y_loader, "prob": p})
        df.to_csv(f, index_label="index")
    assert np.array_equal(df["label"].to_numpy(), y), "test labels out of dataset order"
    return df["prob"].to_numpy()


def score_group(cfg: Config) -> dict | None:
    """Per-seed test metrics, the seed-ensemble (mean probability) and its bootstrap CIs."""
    dirs = seed_dirs(cfg)
    if not dirs:
        print(f"  {cfg.run_name()}: no finished runs, skipped")
        return None
    ev = official(cfg.size)
    y = ev.labels.reshape(-1).astype(int)
    rows, probs = [], []
    for d in dirs:
        p = test_probs(d, cfg.size, y)
        m = compute_metrics(y, p)
        auc, acc = ev.evaluate(p)        # the official MedMNIST numbers must equal ours
        assert abs(auc - m["auc"]) < 1e-9 and abs(acc - m["acc"]) < 1e-9, (d, auc, acc, m)
        save_json(m, d / "test_metrics.json")
        rows.append({"seed": int(d.name[4:]), **m, "val_auc": load_json(d / "metrics.json")["val_auc"]})
        probs.append(p)
    seeds = pd.DataFrame(rows)
    ens_p = np.mean(probs, axis=0)
    info = load_json(dirs[0] / "metrics.json")
    r = {"cfg": cfg, "name": cfg.run_name(), "label": label(cfg), "y": y, "seeds": seeds, "ens_p": ens_p,
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


# ------------------------------------------------------------------------------ Phase 5.1: noise
def noise_eval(cfg: Config) -> pd.DataFrame | None:
    """Evaluate the trained hybrid with a sampled (1,000 shots) or depolarising-noise simulator.

    Only the circuit changes: the trained angles are copied into a QNode built with
    diff_method=None (shots and default.mixed cannot be combined with backprop). No retraining.
    """
    dirs = seed_dirs(cfg)
    if not dirs:
        return None
    y = official(cfg.size).labels.reshape(-1).astype(int)
    per_seed = []
    for d in dirs:
        f = d / "noise_predictions.csv"
        df = pd.read_csv(f) if f.exists() else pd.DataFrame({"label": y, "exact": test_probs(d, cfg.size, y)})
        for setting, kw in NOISE_SETTINGS.items():
            if setting in df:
                continue
            _, model = load_model(d)
            extra = {"device": "default.mixed"} if "noise_p" in kw else {"seed": int(d.name[4:])}
            noisy = make_quantum_layer(cfg.n_qubits, cfg.n_layers, entangle=cfg.entangle, reupload=cfg.reupload,
                                       diff_method=None, **kw, **extra)
            noisy.load_state_dict(model.middle.state_dict())   # same trained angles, different simulator
            model.middle = noisy
            df[setting] = predict(model, test_loader(cfg.size))[0]
            print(f"  noise {d.parent.name}/{d.name} {setting}: AUC {compute_metrics(y, df[setting].to_numpy())['auc']:.4f}")
            df.to_csv(f, index=False)
        per_seed.append(df)
    rows = []
    for setting in ["exact", *NOISE_SETTINGS]:
        ms = pd.DataFrame([compute_metrics(y, df[setting].to_numpy()) for df in per_seed])
        p = NOISE_SETTINGS.get(setting, {}).get("noise_p", 0.0)
        rows.append({"setting": "exact simulation" if setting == "exact" else setting, "p": p,
                     **{f"{m}_{s}": getattr(ms[m], s)() for m in ("auc", "acc") for s in ("mean", "std")}})
    return pd.DataFrame(rows)


def noise_markdown(t: pd.DataFrame, n: int) -> str:
    out = ["| Circuit simulation | Test AUC | Test ACC | Δ AUC vs exact |", "|---|---|---|---|"]
    exact = t.iloc[0]["auc_mean"]
    for _, r in t.iterrows():
        out.append(f"| {r['setting']} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | {r['acc_mean']:.3f} ± {r['acc_std']:.3f} | "
                   f"{r['auc_mean'] - exact:+.4f} |")
    out.append(f"\nSame trained weights, no retraining; mean ± std over {n} seeds. 'shots=1000': each <Z> estimated "
               "from 1,000 measurements. 'p=…': default.mixed simulator with a depolarising channel of probability p "
               "on every qubit after every layer.")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-only", action="store_true", help="only the validation sweep table; never load test")
    ap.add_argument("--noise", action="store_true", help="also run the Phase 5.1 shots/noise evaluation")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (more helps the noisy simulator)")
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
            noise = noise_eval(cfg) if args.noise else pd.read_csv(saved) if saved.exists() else None
            if noise is not None:
                noise.to_csv(saved, index=False)
                figures.noise_curves(noise, FIG_DIR / f"{key}.png", label(cfg))
                sections[key] = noise_markdown(noise, len(seed_dirs(cfg)))

    md = "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items())
    if not args.val_only or not (RUNS_DIR / "results.md").exists():
        (RUNS_DIR / "results.md").write_text("# Generated results (do not edit; rebuilt by src/evaluate.py)\n\n" + md,
                                             encoding="utf-8")
    for k, v in sections.items():
        inject(k, v)
    print(md)


if __name__ == "__main__":
    main()
