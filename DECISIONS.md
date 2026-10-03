# Decisions

Every judgement call made while building this project, with the reason. Newest phase last.

## Phase 0: setup

- **Python 3.13** (`py -3.13`, Microsoft Store build). Neither 3.11 nor 3.12 is installed; the default
  `python` is 3.14, which the brief does not list as tested. 3.13 is listed as tested.
- **Installed versions equal the tested set** (torch 2.14.1+cpu, torchvision 0.29.1+cpu, pennylane 0.45.1,
  pennylane-lightning 0.45.0, medmnist 3.0.2, numpy 2.5.2, scikit-learn 1.9.1), so no API had to be adapted.
- **`requirements.txt` starts with `--extra-index-url https://download.pytorch.org/whl/cpu`**, because the
  frozen `torch==2.14.1+cpu` wheel only exists on PyTorch's index. This keeps the install to one command.
- **Project root is the working folder** (`Quantum-Enhanced-Medical Image Classification/`), not a new
  `qml-pneumonia/` subfolder: the folder was empty and already named for the project.
- **Training runs on the CPU with 2 torch threads per run** (`--threads 2`, the default). Measured on this
  laptop for 2 epochs: 1 thread and 2 threads give bit-identical losses and metrics; 4 threads change the
  4th decimal (different floating-point summation order). Per-epoch time, hybrid / classical:
  1 thread 16 s / 10 s, 2 threads 11.5 s / 6 s, 4 threads 9.3 s / 4.2 s. Two threads is the fastest
  setting that stays bit-identical to single-threaded, and the thread count is saved in every `config.json`.
- **One `PneumoniaNet(mode=...)` class instead of separate `HybridQNN` / `ClassicalControl` classes.** The
  brief's reference implementation uses a `mode` switch; keeping it makes "the two models differ only in the
  middle block" visible in a single constructor.
- **The test split's class counts are asserted in `evaluate.py`, not in `data.py`'s start-up check.**
  `load_split()` asserts the counts of any split it loads; `python src/data.py` only loads train and val,
  so that no script other than `evaluate.py` ever opens the test split.
- **Console output is UTF-8** (`sys.stdout.reconfigure`) in scripts that print circuit drawings, because the
  Windows console's default cp1252 encoding cannot print PennyLane's box-drawing characters.
