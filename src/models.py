"""The models. The two main ones differ ONLY in the 8->8 block in the middle:

    image -> CNN backbone -> 64 features -> Linear(64, 8) -> pi*tanh -> [MIDDLE] -> Linear(8, 1) -> logit
    hybrid:    MIDDLE = 8-qubit, 3-layer circuit       (3 x 8 x 3 = 72 trainable angles)
    classical: MIDDLE = Linear(8, 8) followed by tanh  (8 x 8 + 8 = 72 trainable weights)

Same backbone, same parameter count, same training recipe: any difference in results comes from
the middle block, which is the point of the comparison.
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from quantum import make_quantum_layer


def conv_block(c_in: int, c_out: int) -> list[nn.Module]:
    """3x3 convolution -> batch norm -> ReLU (no conv bias: batch norm has its own shift)."""
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class Backbone(nn.Module):
    """Small CNN: 1 x H x W image -> 64 features. Global average pooling makes it size-agnostic."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(*conv_block(1, 16), *conv_block(16, 32), nn.MaxPool2d(2),
                                      *conv_block(32, 64), nn.MaxPool2d(2), *conv_block(64, 64),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PneumoniaNet(nn.Module):
    """mode='hybrid' uses the quantum circuit as the middle block; mode='classical' uses Linear+tanh."""

    def __init__(self, mode: str = "hybrid", n_qubits: int = 8, n_layers: int = 3, dropout: float = 0.3,
                 entangle: bool = True, reupload: bool = True) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.drop = nn.Dropout(dropout)
        self.pre = nn.Linear(64, n_qubits)
        if mode == "hybrid":
            self.middle = make_quantum_layer(n_qubits, n_layers, entangle=entangle, reupload=reupload)
        elif mode == "classical":
            self.middle = nn.Sequential(nn.Linear(n_qubits, n_qubits), nn.Tanh())
        else:
            raise ValueError(mode)
        self.head = nn.Linear(n_qubits, 1)
        self.probe = self.middle   # the block whose gradient norm train.py logs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squash the features into (-pi, pi): they become rotation angles for the circuit.
        z = math.pi * torch.tanh(self.pre(self.drop(self.backbone(x))))
        # .float(): a float64 tensor reaching a float32 nn.Linear would raise a dtype error.
        z = self.middle(z).float()
        return self.head(z).squeeze(1)   # one logit per image, shape (B,)


class Quanvolution(nn.Module):
    """Trainable quantum filter: a 4-qubit circuit applied to every non-overlapping 2x2 patch.

    Like a 2x2 convolution with stride 2 and 4 output channels, except that the filter is a
    quantum circuit. All patches of all images are folded into one batch for the simulator.
    """

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.qlayer = make_quantum_layer(4, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = F.unfold(x, kernel_size=2, stride=2)                     # (B, 4, h/2 * w/2): one column per patch
        p = p.transpose(1, 2).reshape(-1, 4) * (math.pi / 2)         # (B * patches, 4): pixel values -> angles
        out = self.qlayer(p).float()                                 # (B * patches, 4) expectation values
        return out.reshape(b, h // 2, w // 2, 4).permute(0, 3, 1, 2)  # (B, 4, h/2, w/2) feature maps


class QuanvNet(nn.Module):
    """Phase 5.2: quantum (or classical control) 2x2 front end -> reduced CNN -> logit."""

    def __init__(self, quantum: bool = True, quanv_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        if quantum:
            self.front = Quanvolution(quanv_layers)
        else:
            # Classical control: a 2x2 stride-2 convolution with 4 channels, squashed to [-1, 1] like <Z>.
            self.front = nn.Sequential(nn.Conv2d(1, 4, 2, stride=2), nn.Tanh())
        self.features = nn.Sequential(*conv_block(4, 32), *conv_block(32, 64), nn.MaxPool2d(2),
                                      *conv_block(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(64, 1)
        self.probe = self.front

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(self.front(x)))).squeeze(1)


def build_model(cfg: Config) -> nn.Module:
    """Rebuild any model from its Config (used by train, evaluate, predict and the tests)."""
    if cfg.mode in ("hybrid", "classical"):
        return PneumoniaNet(cfg.mode, cfg.n_qubits, cfg.n_layers, cfg.dropout, cfg.entangle, cfg.reupload)
    return QuanvNet(quantum=cfg.mode == "quanv", quanv_layers=cfg.quanv_layers, dropout=cfg.dropout)


def quantum_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Trainable circuit angles (they get their own learning rate and no weight decay)."""
    return [p for m in model.modules() if isinstance(m, qml.qnn.TorchLayer) for p in m.parameters()]


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Phase 0 check: both main models build, have identical size, and return one logit per image.
    x = torch.randn(4, 1, 28, 28)
    counts = {}
    for mode in ("hybrid", "classical"):
        m = build_model(Config(mode=mode))
        counts[mode] = (count_params(m), count_params(m.middle))
        out = m(x)
        assert out.shape == (4,), out.shape
        print(f"{mode:<10} total {counts[mode][0]:,} params, middle block {counts[mode][1]}, logits {tuple(out.shape)}")
    assert counts["hybrid"] == counts["classical"], counts
    for mode in ("quanv", "quanv_classical"):
        m = build_model(Config(mode=mode))
        print(f"{mode:<16} total {count_params(m):,} params, front {count_params(m.front)}, logits {tuple(m(x).shape)}")
