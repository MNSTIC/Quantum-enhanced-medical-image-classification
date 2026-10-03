"""Smoke tests: run with `pytest` from the project root (about a minute on a laptop CPU).

Each test guards one failure that would silently ruin the experiment: a broken batch dimension,
dead gradients, a checkpoint that does not reload, wrong data counts, or a model that cannot learn.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import Config  # noqa: E402
from data import EXPECTED_COUNTS, load_split, make_train_val_loaders, pos_weight  # noqa: E402
from engine import shift, tta_views  # noqa: E402
from models import build_model, quantum_parameters  # noqa: E402
from quantum import HardwareNoise, make_quantum_layer  # noqa: E402
from train import make_optimizer  # noqa: E402

torch.set_num_threads(2)


def angles(b: int, n: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return math.pi * torch.tanh(torch.randn(b, n))


def test_quantum_layer_shape_dtype_range():
    out = make_quantum_layer(8, 3)(angles(16))
    assert out.shape == (16, 8)
    assert out.dtype == torch.float32
    assert out.min() >= -1.0 and out.max() <= 1.0   # <Z> expectation values


def test_batched_equals_single_samples():
    # Guards the `inputs[i]` mistake: broadcasting over the batch must not mix images.
    layer, x = make_quantum_layer(8, 3), angles(6)
    batched = layer(x)
    single = torch.stack([layer(x[i:i + 1])[0] for i in range(len(x))])
    assert torch.allclose(batched, single, atol=1e-5)


def test_every_quantum_weight_gets_gradient():
    """Every circuit angle that can affect the output gets a non-zero gradient.

    Exception, by physics rather than by bug: the last layer's third Rot angle (omega) is a final
    RZ rotation. Only CNOTs follow it, and CNOTs turn Z measurements into products of Z's, which
    commute with RZ. So those 8 angles cannot change any <Z>; their gradient is exactly 0 up to
    float32 round-off. The circuit therefore has 64 effective angles out of 72 (see DECISIONS.md).
    """
    torch.manual_seed(0)
    model = build_model(Config(mode="hybrid"))
    logits = model(torch.randn(8, 1, 28, 28))
    nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1.] * 4)).backward()
    (w,) = quantum_parameters(model)
    assert w.shape == (3, 8, 3) and w.grad is not None
    g = w.grad.abs()
    inert = g[-1, :, 2]                                        # last layer, omega angles
    active = torch.cat([g[:-1].flatten(), g[-1, :, :2].flatten()])
    assert bool((active > 1e-6).all()), f"dead circuit angle: min |grad| {active.min():.2e}"
    assert bool((inert < 1e-6).all()), f"last-layer omega unexpectedly active: {inert}"


@pytest.mark.parametrize("entangle", [False, True])
def test_one_layer_circuit_has_closed_form(entangle):
    """The final 8-qubit, 1-layer circuit equals a short classical formula (REPORT.md, Section 7).

    Each qubit alone gives c_i = cos(theta_i) cos(x_i) - sin(theta_i) cos(phi_i) sin(x_i).
    The CNOT ring turns the outputs into products: out_0 = c_1 ... c_7, out_k = c_0 ... c_k.
    """
    torch.manual_seed(3)
    layer = make_quantum_layer(8, 1, entangle=entangle)
    with torch.no_grad():
        layer.weights.normal_(0, 1.0)          # large random angles, not just the small initial ones
        x = angles(32)
        out = layer(x)
        phi, theta = layer.weights[0, :, 0], layer.weights[0, :, 1]
        c = torch.cos(theta) * torch.cos(x) - torch.sin(theta) * torch.cos(phi) * torch.sin(x)
    if entangle:
        c = torch.stack([c[:, 1:].prod(1)] + [c[:, :k + 1].prod(1) for k in range(1, 8)], dim=1)
    assert torch.allclose(out, c, atol=1e-5)


@pytest.mark.parametrize("n_layers,entangle", [(1, True), (2, True), (2, False)])
def test_gate_by_gate_noise_circuit_matches_template(n_layers, entangle):
    """The hand-written circuit used for hardware-noise checks equals the template version when the
    noise is zero, and a readout error p_i scales <Z_i> by exactly (1 - 2 p_i)."""
    torch.manual_seed(0)
    exact = make_quantum_layer(4, n_layers, entangle=entangle)
    x = angles(8, 4)
    readout = (0.0, 0.05, 0.1, 0.2)
    with torch.no_grad():
        ref = exact(x)
        for hw, expected in ((HardwareNoise(), ref), (HardwareNoise(readout=readout), ref * (1 - 2 * torch.tensor(readout)))):
            noisy = make_quantum_layer(4, n_layers, entangle=entangle, diff_method=None, device="default.mixed", hw=hw)
            noisy.load_state_dict(exact.state_dict())
            assert torch.allclose(noisy(x).float(), expected, atol=1e-5)


def test_tta_shift_moves_pixels_and_fills_black():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    right = shift(x, 1, 0)
    assert torch.equal(right[..., :, 1:], x[..., :, :-1]) and bool((right[..., :, 0] == -1).all())
    up = shift(x, 0, -1)
    assert torch.equal(up[..., :-1, :], x[..., 1:, :]) and bool((up[..., -1, :] == -1).all())
    assert len(tta_views(x)) == 7 and all(v.shape == x.shape for v in tta_views(x))


@pytest.mark.parametrize("mode", ["hybrid", "classical"])
def test_save_rebuild_load_reproduces_predictions(mode, tmp_path):
    torch.manual_seed(0)
    cfg = Config(mode=mode)
    model, x = build_model(cfg), torch.randn(10, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        before = model(x)
    torch.save(model.state_dict(), tmp_path / "m.pt")

    torch.manual_seed(123)   # different init, so equality can only come from the loaded weights
    rebuilt = build_model(cfg)
    rebuilt.load_state_dict(torch.load(tmp_path / "m.pt"))
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(before, rebuilt(x))


def test_loader_sizes_and_label_counts():
    train_loader, val_loader, pw = make_train_val_loaders(Config())
    for loader, split in ((train_loader, "train"), (val_loader, "val")):
        y = loader.dataset.labels.reshape(-1)
        assert (len(y), (y == 0).sum(), (y == 1).sum()) == EXPECTED_COUNTS[split]
    assert len(train_loader) == 4708 // 64          # drop_last=True
    assert len(val_loader) == math.ceil(524 / 256)
    x, y = next(iter(train_loader))
    assert x.shape == (64, 1, 28, 28) and y.shape == (64, 1)
    assert -1.0 <= x.min() and x.max() <= 1.0       # Normalize(0.5, 0.5) maps [0, 1] to [-1, 1]
    assert abs(pw - 1214 / 3494) < 1e-9


@pytest.mark.parametrize("mode", ["hybrid", "classical"])
def test_overfits_one_fixed_batch(mode):
    """With dropout off, the model must memorise 32 images in 100 steps; if not, something is broken."""
    torch.manual_seed(0)
    ds = load_split("train")                       # no augmentation: the batch is identical every step
    x = torch.stack([ds[i][0] for i in range(32)])
    y = torch.tensor([float(ds[i][1][0]) for i in range(32)])
    cfg = Config(mode=mode, dropout=0.0)
    model = build_model(cfg)
    opt = make_optimizer(model, cfg)               # section 8 optimizer settings, no scheduler
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight(ds.labels)))
    model.train()
    for _ in range(100):
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    with torch.no_grad():
        logits = model(x)
        final_loss = loss_fn(logits, y).item()
    acc = ((logits > 0).float() == y).float().mean().item()
    print(f"{mode}: loss {final_loss:.3f}, acc {acc:.3f}")
    assert acc == 1.0
    assert final_loss < 0.35
