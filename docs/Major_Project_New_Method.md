# Major Project Upgrade: Problem Statement and New Method

## 1) Problem in Current Implementation

The current pipeline has strong results, but it has a research gap that is important for a major project:

- It relies heavily on a single dominant detector path at inference time.
- In real traffic, attack behavior can look normal in one representation (for example reconstruction) but abnormal in another (latent classifier).
- NSL-KDD train/test shift is significant in this project:
  - Train anomaly ratio: 46.54%
  - Test anomaly ratio: 56.92%
  - Train attack types: 22
  - Test attack types: 37

This means a single score threshold is not always enough for robust detection coverage.

## 2) New Method

Name: Calibrated Two-Stage Gated Ensemble

Core idea:

1. Stage-1 detector: Autoencoder reconstruction threshold.
2. Stage-2 detector: Latent classifier probability threshold calibrated from normal-validation traffic.
3. Final decision rule: anomaly if either stage fires (gated OR).

Mathematically:

- Let reconstruction error be $e(x)$ and reconstruction threshold be $\tau_e$.
- Let latent probability be $p(x)$ and latent threshold be $\tau_l$ (calibrated as 95th percentile on normal-validation latent scores).
- Final decision:

$$
\hat{y}(x)=\mathbb{1}\left[e(x)\ge\tau_e \;\;\text{or}\;\; p(x)\ge\tau_l\right]
$$

## 3) Why This Is a Valid Major-Project Contribution

- It is not a bug fix. It is a new algorithmic inference strategy.
- It addresses an identified research problem: detector blind spots under distribution shift.
- It keeps architecture lightweight while improving detection coverage.
- It is production-friendly because calibration is data-driven and simple to maintain.

## 4) Measured Impact on Current Checkpoints

Validation script run on April 10, 2026 in this workspace:

- Baseline AE
  - Accuracy: 0.8620
  - Precision: 0.9366
  - Recall: 0.8125
  - F1: 0.8701
- New Gated Ensemble
  - Accuracy: 0.8713
  - Precision: 0.9203
  - Recall: 0.8473
  - F1: 0.8823

Additional attack-type generalization signals:

- Seen attack recall
  - AE: 0.8158
  - Gated Ensemble: 0.8641
- Unseen attack recall
  - AE: 0.8045
  - Gated Ensemble: 0.8064

Interpretation:

- The new method raises recall and F1 while preserving strong precision.
- It improves coverage especially on seen attacks and slightly improves unseen-attack recall.

## 5) Where It Is Implemented

Notebook implementation:

- notebooks/DL_Anomaly_Detection.ipynb
  - New method explanation section
  - New method training/evaluation section
  - Artifact and inference export for gated predictions

CLI integration:

- main.py
  - test-dl now emits pred_gated_ensemble when gated thresholds are available.

## 6) Suggested Thesis Framing

Suggested title:

"Calibrated Two-Stage Gated Ensemble for Robust Network Anomaly Detection Under Distribution Shift"

Suggested claim:

"Combining reconstruction anomaly evidence with calibrated latent risk signals improves detection coverage and F1 over single-detector deep anomaly baselines on NSL-KDD."

## 7) Publication-Grade Validation Protocol (Added)

To support publication-level evidence (not a single lucky run), use this protocol:

1. Multi-seed stability
- Run the full DL pipeline across multiple random seeds (for example 5 seeds).
- Report per-seed metrics and mean/std for:
  - F1
  - recall
  - seen-attack recall
  - unseen-attack recall
  - normal false-positive rate

2. Paired significance tests
- Compare AE baseline vs gated detector with paired tests on seed-wise results:
  - paired t-test
  - Wilcoxon signed-rank
- Report p-values and effect size direction (delta mean).

3. External-dataset validation
- Evaluate the saved detector on at least one external dataset split (for example UNSW-NB15/CICIDS export) using a documented feature-mapping contract.
- If exact schema mismatch exists, report mapping coverage and missing-feature imputation policy.

4. Claim discipline
- Safe claim: consistent F1/recall gain under NSL-KDD shift.
- Do not overclaim universal zero-day improvement unless external validation confirms it.

Implementation status in this repository:
- `notebooks/DL_Major_Project_Gated_Ensemble.ipynb` now includes:
  - a publication validation block (multi-seed + paired tests)
  - optional external-dataset validation path (auto-skip when no file path is configured)
  - report export wiring (`major_project_report.json`) for publication summaries
