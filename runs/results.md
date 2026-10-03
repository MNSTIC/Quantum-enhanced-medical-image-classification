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

## gradients

| Run group | Runs | Smallest epoch mean | Largest epoch mean | Last epoch (mean over runs) |
|---|---|---|---|---|
| hybrid_q4_l1 | 3 | 4.1e-03 | 3.5e-02 | 2.4e-02 |
| hybrid_q4_l3 | 3 | 1.5e-02 | 6.4e-02 | 3.1e-02 |
| hybrid_q4_l5 | 3 | 1.8e-02 | 5.5e-02 | 4.1e-02 |
| hybrid_q8_l1 | 5 | 1.1e-02 | 6.5e-02 | 3.5e-02 |
| hybrid_q8_l1_e60_ema0.998 | 5 | 1.0e-02 | 7.6e-02 | 3.7e-02 |
| hybrid_q8_l1_noent | 5 | 1.1e-02 | 2.7e-02 | 1.6e-02 |
| hybrid_q8_l1_s64 | 5 | 1.2e-02 | 7.2e-02 | 4.3e-02 |
| hybrid_q8_l3 | 5 | 5.6e-03 | 6.1e-02 | 3.0e-02 |
| hybrid_q8_l3_noent | 3 | 1.7e-02 | 3.5e-02 | 2.2e-02 |
| hybrid_q8_l3_noreup | 3 | 1.6e-02 | 3.6e-02 | 2.3e-02 |
| hybrid_q8_l5 | 3 | 1.0e-02 | 4.5e-02 | 2.2e-02 |

L2 norm of the circuit-angle gradients, averaged over the steps of each epoch (before clipping). A barren plateau would show as values collapsing towards zero.

## phase7_val

| Model | Variant | Seeds | Val AUC | Val loss | Δ val AUC vs v1 | Δ val loss vs v1 |
|---|---|---|---|---|---|---|
| hybrid | v1 | 5 | 0.9956 ± 0.0005 | 0.0600 ± 0.0084 | +0.0000 | +0.0000 |
| hybrid | v1 + TTA | 5 | 0.9970 ± 0.0007 | 0.0515 ± 0.0112 | +0.0014 | -0.0085 |
| hybrid | v2 | 5 | 0.9965 ± 0.0006 | 0.0505 ± 0.0129 | +0.0010 | -0.0095 |
| hybrid | v2 + TTA | 5 | 0.9975 ± 0.0004 | 0.0409 ± 0.0065 | +0.0019 | -0.0191 |
| classical | v1 | 5 | 0.9960 ± 0.0004 | 0.0588 ± 0.0134 | +0.0000 | +0.0000 |
| classical | v1 + TTA | 5 | 0.9971 ± 0.0002 | 0.0442 ± 0.0061 | +0.0011 | -0.0146 |
| classical | v2 | 5 | 0.9967 ± 0.0007 | 0.0486 ± 0.0090 | +0.0006 | -0.0102 |
| classical | v2 + TTA | 5 | 0.9976 ± 0.0004 | 0.0351 ± 0.0030 | +0.0016 | -0.0236 |

Validation only. v1 = the main recipe (30 epochs); v2 = 60 epochs + weight averaging (EMA); TTA = probability averaged over 7 shifted/rotated views. Rule fixed in advance: a candidate is adopted only if, for both the hybrid and the control, it lowers mean validation loss without lowering mean validation AUC.

- **TTA: adopted** by that rule.
- **recipe v2: adopted** by that rule.

## matched

| Model | Params: total / compared block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid 8q x 3L | 61,001 / 72 (64) | 5 | 0.975 ± 0.004 | 0.916 ± 0.023 | 0.984 ± 0.010 | 0.803 ± 0.076 | 0.894 ± 0.033 | 0.980 [0.969, 0.990] | 0.920 [0.897, 0.941] |
| Classical control (8 units) | 61,001 / 72 (72) | 5 | 0.978 ± 0.003 | 0.931 ± 0.016 | 0.966 ± 0.016 | 0.872 ± 0.058 | 0.919 ± 0.024 | 0.981 [0.971, 0.989] | 0.941 [0.921, 0.958] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. Compared block = the 8 -> 8 middle block, or the 2x2 front end for the quanvolution models. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

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
| Quanvolution front end | 5 | 57.0 | 28.5 | 25.0 |
| Conv 2x2 control | 5 | 9.2 | 4.6 | 3.8 |
| Hybrid 8q x 1L, 64x64 | 5 | 105.5 | 52.7 | 43.4 |
| Classical control (8 units), 64x64 | 5 | 78.3 | 39.2 | 28.6 |

CPU only, 2 PyTorch threads per run; up to 8 runs shared the 20-core CPU at once, so single-run times on an idle machine are shorter.

## main

| Model | Params: total / compared block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid 8q x 1L | 60,953 / 24 (16) | 5 | 0.975 ± 0.004 | 0.916 ± 0.028 | 0.971 ± 0.016 | 0.823 ± 0.095 | 0.897 ± 0.041 | 0.979 [0.967, 0.989] | 0.925 [0.904, 0.946] |
| Classical control (8 units) | 61,001 / 72 (72) | 5 | 0.978 ± 0.003 | 0.931 ± 0.016 | 0.966 ± 0.016 | 0.872 ± 0.058 | 0.919 ± 0.024 | 0.981 [0.971, 0.989] | 0.941 [0.921, 0.958] |
| Hybrid 8q x 1L, no entanglement | 60,953 / 24 (16) | 5 | 0.977 ± 0.003 | 0.927 ± 0.010 | 0.975 ± 0.010 | 0.848 ± 0.041 | 0.911 ± 0.016 | 0.981 [0.970, 0.990] | 0.931 [0.909, 0.950] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. Compared block = the 8 -> 8 middle block, or the 2x2 front end for the quanvolution models. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

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

| Model | Params: total / compared block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Quanvolution front end | 56,857 / 24 (20) | 5 | 0.973 ± 0.001 | 0.916 ± 0.017 | 0.929 ± 0.040 | 0.895 ± 0.079 | 0.912 ± 0.026 | 0.976 [0.964, 0.985] | 0.934 [0.915, 0.954] |
| Conv 2x2 control | 56,853 / 20 (20) | 5 | 0.977 ± 0.002 | 0.927 ± 0.009 | 0.934 ± 0.016 | 0.915 ± 0.020 | 0.925 ± 0.009 | 0.979 [0.969, 0.988] | 0.933 [0.913, 0.952] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. Compared block = the 8 -> 8 middle block, or the 2x2 front end for the quanvolution models. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

**A = Quanvolution front end, B = Conv 2x2 control**

| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |
|---|---|---|---|---|---|---|
| AUC | -0.0036 | 0.0023 | [0.964, 0.985] vs [0.969, 0.988] | yes | 1 of 5 | **no significant difference** |
| ACC | -0.0109 | 0.0174 | [0.915, 0.954] vs [0.913, 0.952] | yes | 1 of 5 | **no significant difference** |

## s64

| Model | Params: total / compared block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid 8q x 1L, 64x64 | 60,953 / 24 (16) | 5 | 0.966 ± 0.007 | 0.911 ± 0.022 | 0.964 ± 0.017 | 0.823 ± 0.077 | 0.894 ± 0.032 | 0.974 [0.962, 0.985] | 0.925 [0.904, 0.944] |
| Classical control (8 units), 64x64 | 61,001 / 72 (72) | 5 | 0.973 ± 0.002 | 0.926 ± 0.008 | 0.936 ± 0.013 | 0.908 ± 0.012 | 0.922 ± 0.007 | 0.975 [0.962, 0.985] | 0.933 [0.913, 0.950] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. Compared block = the 8 -> 8 middle block, or the 2x2 front end for the quanvolution models. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

**A = Hybrid 8q x 1L, 64x64, B = Classical control (8 units), 64x64**

| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |
|---|---|---|---|---|---|---|
| AUC | -0.0071 | 0.0072 | [0.962, 0.985] vs [0.962, 0.985] | yes | 0 of 5 | **no significant difference** |
| ACC | -0.0144 | 0.0216 | [0.904, 0.944] vs [0.913, 0.950] | yes | 1 of 5 | **no significant difference** |

## threshold

| Model | Val-tuned threshold | Test ACC at 0.5 | Test ACC at tuned | Sensitivity at tuned | Specificity at tuned | Balanced acc. at tuned | Ensemble ACC at tuned |
|---|---|---|---|---|---|---|---|
| Hybrid 8q x 1L | 0.444 ± 0.331 | 0.916 ± 0.028 | 0.919 ± 0.015 | 0.979 ± 0.007 | 0.818 ± 0.038 | 0.899 ± 0.020 | 0.925 (t = 0.409) |
| Classical control (8 units) | 0.179 ± 0.089 | 0.931 ± 0.016 | 0.904 ± 0.026 | 0.987 ± 0.006 | 0.765 ± 0.074 | 0.876 ± 0.035 | 0.920 (t = 0.257) |
| Hybrid 8q x 1L, no entanglement | 0.337 ± 0.187 | 0.927 ± 0.010 | 0.913 ± 0.015 | 0.986 ± 0.005 | 0.791 ± 0.046 | 0.889 ± 0.021 | 0.925 (t = 0.396) |
| Hybrid 8q x 3L | 0.405 ± 0.210 | 0.916 ± 0.023 | 0.908 ± 0.019 | 0.988 ± 0.009 | 0.775 ± 0.066 | 0.881 ± 0.029 | 0.899 (t = 0.334) |

Secondary result. Each threshold is chosen on validation data only (largest sensitivity + specificity); the test set never influences it. The headline numbers keep the fixed 0.5 threshold.

## phase7

**Post-hoc, exploratory**: the test set had already been read once (Phase 4). These rows are reported whatever the validation verdict, and none of them replaces the headline result.

| Model | Params: total / compared block (effective) | Seeds | Test AUC | Test ACC | Sensitivity | Specificity | Balanced acc. | Ensemble AUC [95% CI] | Ensemble ACC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid 8q x 1L | 60,953 / 24 (16) | 5 | 0.975 ± 0.004 | 0.916 ± 0.028 | 0.971 ± 0.016 | 0.823 ± 0.095 | 0.897 ± 0.041 | 0.979 [0.967, 0.989] | 0.925 [0.904, 0.946] |
| Hybrid 8q x 1L + TTA | 60,953 / 24 (16) | 5 | 0.974 ± 0.002 | 0.891 ± 0.036 | 0.986 ± 0.011 | 0.733 ± 0.112 | 0.859 ± 0.051 | 0.977 [0.964, 0.987] | 0.910 [0.886, 0.933] |
| Hybrid 8q x 1L, 60 epochs, EMA | 60,953 / 24 (16) | 5 | 0.975 ± 0.010 | 0.931 ± 0.014 | 0.968 ± 0.019 | 0.869 ± 0.064 | 0.918 ± 0.024 | 0.980 [0.967, 0.990] | 0.942 [0.923, 0.960] |
| Hybrid 8q x 1L, 60 epochs, EMA + TTA | 60,953 / 24 (16) | 5 | 0.971 ± 0.016 | 0.915 ± 0.028 | 0.981 ± 0.014 | 0.806 ± 0.096 | 0.894 ± 0.041 | 0.976 [0.962, 0.988] | 0.925 [0.904, 0.946] |
| Classical control (8 units) | 61,001 / 72 (72) | 5 | 0.978 ± 0.003 | 0.931 ± 0.016 | 0.966 ± 0.016 | 0.872 ± 0.058 | 0.919 ± 0.024 | 0.981 [0.971, 0.989] | 0.941 [0.921, 0.958] |
| Classical control (8 units) + TTA | 61,001 / 72 (72) | 5 | 0.977 ± 0.003 | 0.915 ± 0.022 | 0.982 ± 0.011 | 0.804 ± 0.070 | 0.893 ± 0.031 | 0.980 [0.970, 0.989] | 0.918 [0.896, 0.939] |
| Classical control (8 units), 60 epochs, EMA | 61,001 / 72 (72) | 5 | 0.982 ± 0.002 | 0.941 ± 0.006 | 0.971 ± 0.012 | 0.891 ± 0.023 | 0.931 ± 0.008 | 0.984 [0.975, 0.992] | 0.949 [0.931, 0.965] |
| Classical control (8 units), 60 epochs, EMA + TTA | 61,001 / 72 (72) | 5 | 0.981 ± 0.002 | 0.927 ± 0.010 | 0.986 ± 0.007 | 0.830 ± 0.032 | 0.908 ± 0.014 | 0.982 [0.972, 0.991] | 0.929 [0.909, 0.949] |

Mean ± sample standard deviation over seeds; ACC, sensitivity and specificity at threshold 0.5. Ensemble = average of the seeds' probabilities; CI = 2,000 bootstrap resamples of the 624 test images. Compared block = the 8 -> 8 middle block, or the 2x2 front end for the quanvolution models. 'Effective' excludes the last-layer RZ angles that provably cannot affect the circuit output.

Hybrid vs classical control with the variant chosen on validation data (v2 + TTA):

**A = Hybrid 8q x 1L, 60 epochs, EMA + TTA, B = Classical control (8 units), 60 epochs, EMA + TTA**

| Metric | Mean A − B | Larger seed std | Ensemble 95% CI, A vs B | CIs overlap | Seeds where A > B | Verdict |
|---|---|---|---|---|---|---|
| AUC | -0.0104 | 0.0163 | [0.962, 0.988] vs [0.972, 0.991] | yes | 0 of 5 | **no significant difference** |
| ACC | -0.0119 | 0.0280 | [0.904, 0.946] vs [0.909, 0.949] | yes | 2 of 5 | **no significant difference** |

## devices

| Model | Device | CPU threads | PyTorch build | Seconds per epoch |
|---|---|---|---|---|
| hybrid (final, 8q x 1L) | cpu | 2 | 2.14.1+cu130 | 5.52 |
| classical control | cpu | 2 | 2.14.1+cu130 | 3.77 |
| hybrid (final, 8q x 1L) | cpu | 8 | 2.14.1+cu130 | 4.18 |
| classical control | cpu | 8 | 2.14.1+cu130 | 2.40 |
| hybrid (final, 8q x 1L) | cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | 2 | 2.14.1+cu130 | 3.69 |
| classical control | cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | 2 | 2.14.1+cu130 | 1.06 |
| hybrid (final, 8q x 1L) | cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | 8 | 2.14.1+cu130 | 3.57 |
| classical control | cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | 8 | 2.14.1+cu130 | 1.08 |

One training epoch plus one validation pass, mean of epochs 2-3 (epoch 1 is warm-up), one run at a time. Every reported result was trained on the CPU with 2 threads, which is bit-for-bit reproducible; GPU arithmetic differs slightly.