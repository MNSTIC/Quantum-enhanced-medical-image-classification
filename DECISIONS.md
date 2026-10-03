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

## Phase 1-2: first runs and tests

- **8 of the 72 circuit angles cannot affect the output, and the circuit is kept as specified anyway.**
  The smoke test "every quantum weight receives a non-zero gradient" failed on exactly the last layer's
  third `Rot` angle (omega) on each qubit: gradients of order 1e-8 (float32 round-off) against at least
  5e-4 for every other angle, for 1, 3 and 5 layers, with and without entanglement. The reason is exact:
  `Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)`, so the last thing each qubit sees before the
  CNOT ring is `RZ(omega)`. CNOTs map Z-type observables to products of Z's, and those commute with RZ,
  so `<Z_i>` does not depend on omega. The brief says to keep the reference circuit's behaviour, so the
  circuit is unchanged; instead (a) the test asserts both facts explicitly, and (b) every results table
  reports the middle block as "72 (64 effective)". This matters for fairness: the hybrid's middle block
  has 64 working parameters against the classical control's 72. The same holds for the 4-qubit
  quanvolution filter (24 angles, 20 effective), whose classical control conv also has 20.
- **Overfit test thresholds pass with margin**: after 100 steps on a fixed batch of 32, hybrid loss 0.141,
  classical 0.059, both 100% accuracy (the brief's prototype saw 0.25 and 0.12).
- **Validation accuracy at threshold 0.5 sits below what the AUC suggests** (classical seed 0: AUC 0.996,
  ACC 0.910, specificity 1.000, sensitivity 0.879). `pos_weight` makes the loss treat both classes equally
  although 74% of training images are pneumonia, so the model is cautious about calling pneumonia at 0.5.
  The brief fixes 0.5 as the headline threshold; this is reported, not tuned away.
- **`evaluate.py` caches each checkpoint's test probabilities** (`test_predictions.csv`) and reuses them, so
  rebuilding tables and figures later never re-scores a model. It also asserts that the official MedMNIST
  `Evaluator` returns the same AUC and ACC as our scikit-learn code (to 1e-9), that labels are in dataset
  order, and prints a warning if test ACC exceeds 0.96 or test AUC exceeds validation AUC.
- **Comparison rule, made concrete**: "the seed standard deviation" is taken as the *larger* of the two
  models' seed standard deviations (the conservative reading); the rule then says "no significant
  difference" if the mean difference is smaller than that, or if the two ensembles' 95% bootstrap CIs
  overlap. The number of seeds on which each model wins (paired by seed) is reported alongside.
- **Results tables are injected into README.md and REPORT.md** between `<!-- results:KEY -->` markers by
  `evaluate.py`, so no number in those documents is typed by hand.

## Phase 3: sweep

- **Runs execute in parallel (8 `train.py` processes, 2 threads each), not one after another.** The brief asks
  for a sequential, resumable sweep to keep it simple and restartable. Each job here is still an independent
  process with its own seed and fixed thread count, so the numbers are identical to a sequential run; the
  sweep still skips finished jobs (`metrics.json` present) and resumes after an interruption. Parallelism
  only shortens wall-clock time. Measured epoch times in the parallel sweep are inflated by contention
  (another application was also using about 6 of the 20 cores), so timing tables carry that caveat.
- **Final hybrid: 8 qubits x 1 layer** (`FINAL_HYBRID` in `src/config.py`), chosen on validation data only,
  before any test scoring. Every setting landed between 0.9943 and 0.9959 mean validation AUC with seed standard
  deviations around 0.001, so on AUC they are statistically tied. The brief says to prefer the smaller circuit
  when settings tie, so "tie" was made concrete with the same rule used for the final comparison (difference
  in means smaller than the larger seed std), applied to both criteria the brief names, validation AUC and
  validation loss, against the best size-grid setting on each:
  - 4q x 1L ties on AUC but not on loss (0.0679 ± 0.0122 against 0.0545 ± 0.0021) and is the least stable
    setting; excluded.
  - 8q x 1L ties on both. It is the smallest remaining circuit by trainable angles (24), gates (24),
    two-qubit gates (8) and depth. 4q x 3L also ties and uses fewer qubits, but has more angles (36), gates
    (36), CNOTs (12) and depth, so on hardware it would be the larger and noisier circuit. Hence 8q x 1L.
  - With one layer, re-uploading has no effect (there is only one encoding).
- **Consequence: the final hybrid is not parameter-matched.** Its middle block has 24 angles (16 effective)
  against the classical control's 72 weights; the totals are 60,953 and 61,001. The results tables report
  both. Because the brief's central design is a 72-vs-72 comparison, the default 8q x 3L circuit is
  **also trained with 5 seeds and compared with the same classical control as a secondary, parameter-matched
  comparison** (`MATCHED_HYBRID`, `--stage matched`). This was declared here before the test set was read.
- **The no-entanglement circuit had the lowest validation loss** (0.0470 ± 0.0022 against about 0.055 for
  every entangled setting, a gap of over 2 standard deviations) at equal validation AUC. It was not chosen as
  the final hybrid because the brief's design makes entanglement the default and runs its removal as the
  ablation; the observation is reported in REPORT.md and tested again by the Phase 4 ablation.
- **The classical backbone was not swept**, as the brief requires: changing it would change what is compared.
