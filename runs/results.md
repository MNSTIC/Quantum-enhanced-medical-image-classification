# Generated results (do not edit; rebuilt by src/evaluate.py)

## sweep

| Setting | Seeds | Circuit params | Val AUC | Val loss | Val ACC | Mean best epoch | Epoch time (s) |
|---|---|---|---|---|---|---|---|
| 4q x 1L | 3 | 12 | 0.9943 ± 0.0016 | 0.0679 ± 0.0122 | 0.956 | 26.0 | 45.4 |
| 4q x 3L | 3 | 36 | 0.9955 ± 0.0010 | 0.0547 ± 0.0019 | 0.962 | 21.0 | 48.1 |
| 4q x 5L | 3 | 60 | 0.9956 ± 0.0002 | 0.0556 ± 0.0061 | 0.965 | 23.0 | 41.0 |
| 8q x 1L | 3 | 24 | 0.9957 ± 0.0006 | 0.0559 ± 0.0087 | 0.964 | 19.3 | 20.7 |
| 8q x 3L | 3 | 72 | 0.9958 ± 0.0010 | 0.0553 ± 0.0035 | 0.971 | 20.7 | 23.5 |
| 8q x 5L | 3 | 120 | 0.9954 ± 0.0010 | 0.0545 ± 0.0021 | 0.961 | 19.3 | 34.9 |
| 8q x 3L, no entanglement | 3 | 72 | 0.9957 ± 0.0007 | 0.0470 ± 0.0022 | 0.964 | 22.0 | 26.3 |
| 8q x 3L, no re-uploading | 3 | 72 | 0.9959 ± 0.0009 | 0.0535 ± 0.0057 | 0.961 | 20.3 | 16.6 |

Validation split only (524 images); mean ± sample std over seeds 0-2. Checkpoint = epoch with the best validation AUC. Epoch times measured with up to 8 runs sharing the CPU.

## matched

| Model | Params: total / middle block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid 8q x 3L | 61,001 / 72 (64) | 5 | 0.975 ± 0.004 | 0.916 ± 0.023 | 0.984 ± 0.010 | 0.803 ± 0.076 | 0.894 ± 0.033 | 0.980 [0.969, 0.990] | 0.920 [0.897, 0.941] |
| Classical control (8 units) | 61,001 / 72 (72) | 5 | 0.978 ± 0.003 | 0.931 ± 0.016 | 0.966 ± 0.016 | 0.872 ± 0.058 | 0.919 ± 0.024 | 0.981 [0.971, 0.989] | 0.941 [0.921, 0.958] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

**A = Hybrid 8q x 3L, B = Classical control (8 units)**

| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |
|---|---|---|---|---|---|---|
| AUC | -0.0026 | 0.0040 | [0.969, 0.990] vs [0.971, 0.989] | yes | 2 of 5 | **no significant difference** |
| ACC | -0.0144 | 0.0225 | [0.897, 0.941] vs [0.921, 0.958] | yes | 1 of 5 | **no significant difference** |

## timing

| Model | Seeds | Mean epoch time (s) | Mean training run (min, 30 epochs) | Fastest run (min) |
|---|---|---|---|---|
| Hybrid 8q x 1L | 5 | 19.1 | 9.6 | 8.3 |
| Classical control (8 units) | 5 | 11.7 | 5.8 | 3.6 |
| Hybrid 8q x 1L, no entanglement | 5 | 17.1 | 8.5 | 8.2 |
| Hybrid 8q x 3L | 5 | 24.6 | 12.3 | 6.6 |
| Conv 2x2 control | 3 | 7.9 | 3.9 | 3.8 |

CPU only, 2 PyTorch threads per run; up to 8 runs shared the 20-core CPU at once, so single-run times on an idle machine are shorter.

## main

| Model | Params: total / middle block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid 8q x 1L | 60,953 / 24 (16) | 5 | 0.975 ± 0.004 | 0.916 ± 0.028 | 0.971 ± 0.016 | 0.823 ± 0.095 | 0.897 ± 0.041 | 0.979 [0.967, 0.989] | 0.925 [0.904, 0.946] |
| Classical control (8 units) | 61,001 / 72 (72) | 5 | 0.978 ± 0.003 | 0.931 ± 0.016 | 0.966 ± 0.016 | 0.872 ± 0.058 | 0.919 ± 0.024 | 0.981 [0.971, 0.989] | 0.941 [0.921, 0.958] |
| Hybrid 8q x 1L, no entanglement | 60,953 / 24 (16) | 5 | 0.977 ± 0.003 | 0.927 ± 0.010 | 0.975 ± 0.010 | 0.848 ± 0.041 | 0.911 ± 0.016 | 0.981 [0.970, 0.990] | 0.931 [0.909, 0.950] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

## seeds

| Model | Metric | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|---|
| Hybrid 8q x 1L | val AUC | 0.995 | 0.996 | 0.996 | 0.995 | 0.996 |
| Hybrid 8q x 1L | test AUC | 0.971 | 0.976 | 0.971 | 0.980 | 0.975 |
| Hybrid 8q x 1L | test ACC | 0.897 | 0.925 | 0.877 | 0.946 | 0.934 |
| Hybrid 8q x 1L | test sensitivity | 0.969 | 0.985 | 0.990 | 0.962 | 0.951 |
| Hybrid 8q x 1L | test specificity | 0.778 | 0.825 | 0.688 | 0.919 | 0.906 |
| Classical control (8 units) | val AUC | 0.996 | 0.996 | 0.996 | 0.996 | 0.996 |
| Classical control (8 units) | test AUC | 0.975 | 0.978 | 0.974 | 0.981 | 0.981 |
| Classical control (8 units) | test ACC | 0.925 | 0.942 | 0.905 | 0.944 | 0.938 |
| Classical control (8 units) | test sensitivity | 0.946 | 0.959 | 0.987 | 0.977 | 0.962 |
| Classical control (8 units) | test specificity | 0.889 | 0.915 | 0.769 | 0.889 | 0.897 |
| Hybrid 8q x 1L, no entanglement | val AUC | 0.997 | 0.997 | 0.996 | 0.996 | 0.995 |
| Hybrid 8q x 1L, no entanglement | test AUC | 0.980 | 0.974 | 0.975 | 0.975 | 0.981 |
| Hybrid 8q x 1L, no entanglement | test ACC | 0.928 | 0.929 | 0.912 | 0.928 | 0.939 |
| Hybrid 8q x 1L, no entanglement | test sensitivity | 0.977 | 0.979 | 0.985 | 0.974 | 0.959 |
| Hybrid 8q x 1L, no entanglement | test specificity | 0.846 | 0.846 | 0.791 | 0.850 | 0.906 |

## target

Target (beat ResNet-18 at 28x28, AUC 0.944 / ACC 0.854): Hybrid 8q x 1L mean test AUC 0.975 (+0.031), ACC 0.916 (+0.062) over 5 seeds: **met**.

## baselines

| Published method (MedMNIST v2, Table 3) | Test AUC | Test ACC |
|---|---|---|
| ResNet-18 (28x28) | 0.944 | 0.854 |
| ResNet-18 (224x224) | 0.956 | 0.864 |
| ResNet-50 (28x28) | 0.948 | 0.854 |
| ResNet-50 (224x224) | 0.962 | 0.884 |
| auto-sklearn | 0.942 | 0.855 |
| AutoKeras | 0.947 | 0.878 |
| Google AutoML Vision | 0.991 | 0.946 |

## compare

**A = Hybrid 8q x 1L, B = Classical control (8 units)**

| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |
|---|---|---|---|---|---|---|
| AUC | -0.0031 | 0.0036 | [0.967, 0.989] vs [0.971, 0.989] | yes | 0 of 5 | **no significant difference** |
| ACC | -0.0151 | 0.0282 | [0.904, 0.946] vs [0.921, 0.958] | yes | 1 of 5 | **no significant difference** |

**A = Hybrid 8q x 1L, B = Hybrid 8q x 1L, no entanglement**

| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |
|---|---|---|---|---|---|---|
| AUC | -0.0022 | 0.0036 | [0.967, 0.989] vs [0.970, 0.990] | yes | 2 of 5 | **no significant difference** |
| ACC | -0.0115 | 0.0282 | [0.904, 0.946] vs [0.909, 0.950] | yes | 1 of 5 | **no significant difference** |

## quanv

| Model | Params: total / middle block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Conv 2x2 control | 56,853 / 20 (20) | 3 | 0.978 ± 0.000 | 0.932 ± 0.008 | 0.946 ± 0.003 | 0.909 ± 0.021 | 0.927 ± 0.010 | 0.980 [0.970, 0.988] | 0.941 [0.921, 0.958] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.