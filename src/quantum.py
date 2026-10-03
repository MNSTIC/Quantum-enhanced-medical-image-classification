"""The variational quantum circuit, wrapped as a PyTorch layer.

Picture the circuit as a tiny neural layer whose "neurons" are qubits:
  1. encode: each input number becomes the rotation angle of one qubit (angle encoding);
  2. process: trainable rotations on every qubit, then CNOT gates that entangle neighbours;
  3. read out: measure the expectation value <Z> of every qubit, a number in [-1, 1].
Steps 1-2 repeat n_layers times ("data re-uploading"). The simulator computes the exact
quantum state, so PyTorch can backpropagate through every gate like through any other layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pennylane as qml
import torch
import torch.nn as nn


@dataclass(frozen=True)
class HardwareNoise:
    """A simple, gate-by-gate noise model for the Phase 5.1 / 7 robustness checks (evaluation only).

    p1:      depolarising probability after every single-qubit gate (RY encoding and Rot)
    p2:      depolarising probability on BOTH qubits after every CNOT (two-qubit gates are the noisiest)
    gamma:   amplitude damping (energy loss, T1) and phase damping (dephasing, T2) per qubit per layer
    readout: per-qubit probability that the measured bit is flipped (readout error)
    These are order-of-magnitude stand-ins for a real device, not a calibrated model of one.
    """
    p1: float = 0.0
    p2: float = 0.0
    gamma: float = 0.0
    readout: tuple[float, ...] = ()


def _noisy_circuit(inputs, weights, n_qubits: int, n_layers: int, entangle: bool, reupload: bool,
                   hw: HardwareNoise) -> None:
    """The same gates as AngleEmbedding + StronglyEntanglingLayers, written out one by one so that
    noise can follow each gate. tests/test_smoke.py checks that with zero noise it equals the template."""
    def depolarise(p, wires):
        if p > 0:
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)

    for layer in range(n_layers):
        if reupload or layer == 0:
            for w in range(n_qubits):
                qml.RY(inputs[..., w], wires=w)   # inputs[..., w] keeps the batch dimension intact
                depolarise(hw.p1, [w])
        for w in range(n_qubits):
            qml.Rot(weights[layer, w, 0], weights[layer, w, 1], weights[layer, w, 2], wires=w)
            depolarise(hw.p1, [w])
        if entangle:
            for w in range(n_qubits):   # the ring StronglyEntanglingLayers uses: w -> w + 1, last -> first
                qml.CNOT(wires=[w, (w + 1) % n_qubits])
                depolarise(hw.p2, [w, (w + 1) % n_qubits])
        if hw.gamma > 0:
            for w in range(n_qubits):
                qml.AmplitudeDamping(hw.gamma, wires=w)
                qml.PhaseDamping(hw.gamma, wires=w)
    for w, p in enumerate(hw.readout):
        if p > 0:
            qml.BitFlip(p, wires=w)   # a flipped measurement outcome, applied just before measuring Z


def make_qnode(n_qubits: int, n_layers: int, entangle: bool = True, reupload: bool = True,
               diff_method: str | None = "backprop", device: str = "default.qubit",
               noise_p: float = 0.0, shots: int | None = None, seed: int | None = None,
               hw: HardwareNoise | None = None) -> qml.QNode:
    """Build the circuit as a QNode (a quantum function bound to a simulator).

    Training uses the defaults: exact state-vector simulation with backpropagation.
    Phase 5 robustness checks reuse the same circuit with finite `shots` (sampling noise, like
    a real device) or on `default.mixed` with depolarising noise of strength `noise_p`.
    Both of those need diff_method=None: they are evaluated only, never trained.
    """
    dev = qml.device(device, wires=n_qubits, seed=seed)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if getattr(inputs, "is_cuda", False):
            # GPU only: PennyLane builds the starting state |0...0> on the CPU, which clashes with GPU
            # inputs, so prepare it on the inputs' device. Skipped on the CPU, so CPU results are unchanged.
            start = torch.zeros(2 ** n_qubits, dtype=torch.complex64, device=inputs.device)
            start[0] = 1
            qml.StatePrep(start, wires=range(n_qubits))
        if hw is not None:   # gate-by-gate noise model (needs device="default.mixed")
            _noisy_circuit(inputs, weights, n_qubits, n_layers, entangle, reupload, hw)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        # `inputs` has shape (batch, n_qubits). Templates such as AngleEmbedding treat the leading
        # dimension as a batch and simulate all images at once ("parameter broadcasting").
        # Never write inputs[i] here: that would pick image i, not feature i.
        for layer in range(n_layers):
            if reupload or layer == 0:
                # ENCODING: RY(x_i) on qubit i turns feature i into a rotation angle.
                # Re-uploading repeats this before every layer, which lets the circuit build
                # higher-frequency functions of the inputs (more expressive).
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            if entangle:
                # TRAINABLE LAYER: Rot(phi, theta, omega) on each qubit, then a ring of CNOTs
                # (0->1, 1->2, ..., n-1->0). The CNOTs create entanglement, so the measured value
                # of one qubit can depend on all inputs, not only on its own.
                qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(n_qubits))
            else:
                # ABLATION: identical trainable rotations but no CNOTs. Each qubit then evolves
                # on its own and output i depends on input i only.
                for w in range(n_qubits):
                    qml.Rot(weights[layer, w, 0], weights[layer, w, 1], weights[layer, w, 2], wires=w)
            if noise_p > 0:
                # NOISE MODEL (evaluation only): after each layer every qubit is replaced by the
                # maximally mixed state with probability p, a crude stand-in for hardware errors.
                for w in range(n_qubits):
                    qml.DepolarizingChannel(noise_p, wires=w)
        # MEASUREMENT: expectation value of Pauli-Z on each qubit, i.e. P(0) - P(1), in [-1, 1].
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    if shots is not None:
        circuit = qml.set_shots(circuit, shots=shots)   # only allowed when diff_method is None
    return circuit


def weight_shape(n_qubits: int, n_layers: int) -> tuple[int, ...]:
    """(n_layers, n_qubits, 3): three Rot angles per qubit per layer."""
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


def small_angle_init(t: torch.Tensor) -> torch.Tensor:
    """Normal(0, 0.1) initial angles.

    PennyLane's default (uniform on [0, 2pi]) starts the circuit at a random unitary, where
    gradients are exponentially small (a "barren plateau"). Small angles start it close to the
    identity, where gradients are large.
    """
    return nn.init.normal_(t, mean=0.0, std=0.1)


def make_quantum_layer(n_qubits: int, n_layers: int, **qnode_kwargs) -> qml.qnn.TorchLayer:
    """The circuit as an nn.Module: input (B, n_qubits) angles -> output (B, n_qubits) in [-1, 1].

    TorchLayer registers `weights` as a normal trainable torch Parameter.
    """
    qnode = make_qnode(n_qubits, n_layers, **qnode_kwargs)
    return qml.qnn.TorchLayer(qnode, {"weights": weight_shape(n_qubits, n_layers)},
                              init_method=small_angle_init)


def draw_circuit(n_qubits: int, n_layers: int, path, entangle: bool = True, reupload: bool = True) -> str:
    """Save the decomposed circuit (as the simulator runs it) to `path` and return a text drawing."""
    import matplotlib.pyplot as plt

    qnode = make_qnode(n_qubits, n_layers, entangle=entangle, reupload=reupload)
    x = torch.full((n_qubits,), 0.5)
    w = torch.full(weight_shape(n_qubits, n_layers), 0.1)
    fig, _ = qml.draw_mpl(qnode, level="device", decimals=None)(x, w)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return qml.draw(qnode, level="device", decimals=None)(x, w)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")   # circuit drawings use box characters; Windows consoles default to cp1252
    layer = make_quantum_layer(8, 3)
    x = math.pi * torch.tanh(torch.randn(5, 8))
    y = layer(x)
    print("output", tuple(y.shape), y.dtype, f"range [{y.min():.3f}, {y.max():.3f}]")
    print(qml.draw(make_qnode(4, 1), level="device", decimals=None)(torch.zeros(4), torch.zeros(1, 4, 3)))
