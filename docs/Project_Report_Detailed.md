# Project Report: Network Anomaly Detection (NSL-KDD)

## 1) Executive Summary

This project builds and evaluates a network anomaly detection system on NSL-KDD. It establishes strong classical ML baselines, a deep learning autoencoder baseline, and proposes a new calibrated two-stage gated ensemble that improves detection coverage under train/test shift. A production-grade hybrid pipeline and a publication-grade evaluation protocol are also delivered to support deployment and rigorous reporting.

## 2) Problem Statement and Motivation

Network intrusion data often shifts between training and test distributions. On NSL-KDD, the anomaly ratio increases from 46.54% (train) to 56.92% (test), and the attack type count expands from 22 to 37. Under such shifts, a single detector can miss attacks that look normal in one representation but abnormal in another. The core goal is to improve recall and F1 without sacrificing operational simplicity.

## 3) Dataset and Task Definition

- Dataset: NSL-KDD (KDDTrain+.txt, KDDTest+.txt)
- Task: binary classification (normal = 0, attack = 1)
- Feature types: numeric and categorical
- Train/test shift: increased anomaly ratio and more attack types in test

## 4) Preprocessing and Feature Handling

- Categorical features are encoded with an ordinal encoder.
- Numeric features are imputed (median) and scaled (MinMax or Standard scaling, depending on pipeline).
- Train and test are aligned to common columns to avoid schema mismatch.
- Preprocessing artifacts are persisted for reproducible inference (encoder, scaler, feature names, categorical list).

## 5) Classical ML Baselines

The ML baseline suite is fully implemented and evaluated using unsupervised anomaly detectors trained on normal traffic:

- Isolation Forest
- Local Outlier Factor (novelty mode)
- One-Class SVM

Evaluation includes ROC/PR curves, confusion matrices, calibrated thresholds, and attack-type recall analysis.

### ML Results (current run snapshot)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.8049 | 0.9718 | 0.6769 | 0.7980 | 0.9371 | 0.9527 |
| Local Outlier Factor | 0.7661 | 0.8246 | 0.7483 | 0.7846 | 0.8547 | 0.8228 |
| One-Class SVM | 0.7838 | 0.9203 | 0.6790 | 0.7814 | 0.8670 | 0.8765 |

Interpretation: ML baselines are fast and precise, but recall can lag under distribution shift.

## 6) Deep Learning Baseline

### Autoencoder Reconstruction Detector

- Autoencoder is trained on normal traffic only.
- Reconstruction error is computed on test data.
- Thresholding on reconstruction error yields anomaly predictions.

This baseline typically improves recall and F1 compared to classical ML, but can still miss attacks that reconstruct well.

## 7) Proposed Method (Major Contribution)

### Calibrated Two-Stage Gated Ensemble

Motivation: A single detector misses attacks that are abnormal in one representation but normal in another. The proposed method combines both signals in a calibrated and lightweight ensemble.

**Pipeline:**
1. Stage 1 detector: Autoencoder reconstruction error threshold.
2. Stage 2 detector: Latent-space Logistic Regression classifier.
3. Calibration: Latent probability threshold is set from normal-validation samples (95th percentile).
4. Final decision: Gated OR (anomaly if either stage triggers).

**Decision rule:**

$\hat{y}(x)=\mathbb{1}[e(x) \ge \tau_e \;\text{or}\; p(x) \ge \tau_l]$

### Measured Impact (current run snapshot)

| Detector | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline AE | 0.8620 | 0.9366 | 0.8125 | 0.8701 |
| Gated Ensemble | 0.8713 | 0.9203 | 0.8473 | 0.8823 |

Additional attack-type recall indicators:
- Seen-attack recall improves (AE 0.8158 to Gated 0.8641)
- Unseen-attack recall slightly improves (AE 0.8045 to Gated 0.8064)

Interpretation: The gated ensemble improves recall and F1 with minimal architectural complexity and clear operational semantics.

## 8) Production-Grade Hybrid Pipeline

A production pipeline is implemented for deployment and monitoring with calibrated, explainable outputs:

**Components:**
- Denoising autoencoder (reconstruction signal)
- Latent Logistic Regression classifier
- Latent Isolation Forest detector
- Isotonic calibration for component scores
- Stacked meta-classifier over calibrated probabilities
- Tuned threshold (F1-optimized on calibration set)
- Conformal threshold (normal-quantile based)

**Outputs:**
- Tuned and conformal predictions
- Combined anomaly probability
- Dominant detector
- Risk tier
- Metadata and monitoring baseline JSON

This pipeline provides higher robustness and clearer deployment controls compared to single-model baselines.

## 9) Evaluation Protocol (Publication-Grade)

To avoid single-run bias, a publication workflow is provided:

1. Multi-seed stability (5+ seeds)
2. Paired significance tests (t-test and Wilcoxon)
3. Attack-type recall breakdowns
4. Optional external dataset evaluation with a feature mapping contract

This protocol supports statistically grounded claims rather than single-run results.

## 9.1) Results and Comparison (Report-Ready)

### A) Baseline vs Proposed (DL)

The proposed gated ensemble is better than the single-detector baseline because it reduces blind spots. In the current run snapshot, it improves recall and F1 while keeping precision strong.

| Detector | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline AE | 0.8620 | 0.9366 | 0.8125 | 0.8701 |
| Gated Ensemble | 0.8713 | 0.9203 | 0.8473 | 0.8823 |

**Why it is better than the previous method:**
- The AE baseline uses only reconstruction error; the gated method adds a calibrated latent classifier.
- Attacks that reconstruct well but look abnormal in latent space are recovered by the gated OR rule.
- Seen-attack recall improves (0.8158 to 0.8641) while unseen-attack recall is maintained (0.8045 to 0.8064).

### B) ML Baseline Context (for completeness)

The best ML baseline (Isolation Forest) is precise but has lower recall under shift.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.8049 | 0.9718 | 0.6769 | 0.7980 | 0.9371 | 0.9527 |

### C) Production Pipeline (example run snapshot)

The production hybrid pipeline adds calibrated stacking for deployment. Example run metrics from the implementation summary:

- Tuned threshold F1: 0.7599
- Conformal threshold F1: 0.8045
- ROC-AUC: 0.9379

These values are run-dependent; re-run for final report numbers.

## 9.2) Attached Figures (Available in Repo)

The report can embed these existing images from the repository:

![Autoencoder Architecture](../extra/autoencoder-net-arch.png)

![Autoencoder Results](../extra/autoencoder-results.png)

![Correlation Matrix](../extra/Corr_matrix.png)

![t-SNE Projection 1](../extra/TSNE-1.png)

![t-SNE Embeddings](../extra/TSNE-embeddings.png)

![Packet Sent Analysis](../extra/packetSent.png)

If you want the full results package, I can generate and add the missing figures listed in the checklist (ROC, PR, confusion matrices, attack-type recall, multi-seed tables) and then embed them here.

## 10) Reproducibility and CLI Workflow

Key CLI commands:

```bash
# Download NSL-KDD
python main.py download-data --data_dir ./data

# Train ML
python main.py train-ml --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/ml_models_major

# Train DL (AE + latent)
python main.py train-dl --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major

# Test ML
python main.py test-ml --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/ml_models_major --output_csv ./ml_test_predictions.csv

# Test DL
python main.py test-dl --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/autoencoder_major --output_csv ./dl_test_predictions.csv

# Publication evaluation (multi-seed)
python main.py eval-publication --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major_project_gated --seeds 7,21,42,84,126 --sample_fraction 1.0 --epochs 30 --target_fprs 0.01,0.05,0.10
```

## 11) Artifacts and Outputs

### ML artifacts
- checkpoints/ml_models_major/
  - isolation_forest.joblib
  - local_outlier_factor.joblib
  - oneclass_svm.joblib
  - standard_scaler.joblib
  - ordinal_encoder.joblib
  - feature_names.joblib
  - categorical_columns.joblib
  - ml_metadata.json

### DL artifacts
- checkpoints/autoencoder_major/
  - autoencoder_major.keras
  - latent_classifier.joblib
  - standard_scaler.joblib
  - ordinal_encoder.joblib
  - feature_names.joblib
  - categorical_columns.joblib
  - dl_metadata.json

### Inference outputs
- ml_test_predictions.csv
- dl_test_predictions.csv
- ml_vs_dl_predictions.csv

### Production artifacts
- checkpoints/production_hybrid/
  - prod_autoencoder.keras
  - prod_latent_logreg.joblib
  - prod_latent_isolation_forest.joblib
  - prod_meta_classifier.joblib
  - calibration artifacts
  - prod_thresholds.json
  - prod_metadata.json
  - prod_monitor_baseline.json

## 12) Limitations and Threats to Validity

- NSL-KDD is a benchmark dataset with known artifacts and may not reflect live enterprise traffic.
- Threshold calibration depends on normal-validation data quality.
- External validation is optional and must be performed before making broad generalization claims.

## 13) Future Work

- Validate on external datasets (UNSW-NB15 or CICIDS) with documented feature mapping.
- Explore lightweight conformal calibration across multiple operating points.
- Add streaming evaluation for online drift monitoring.

## 14) Conclusion

This project delivers a complete anomaly detection stack: strong ML baselines, a robust DL baseline, a new calibrated gated ensemble method that improves recall and F1 under distribution shift, and a production-grade hybrid pipeline with calibrated, deployment-ready outputs. The evaluation protocol supports reproducible, statistically grounded reporting.

## 15) Figures and Tables for the Report (Detailed Checklist)

This section lists the recommended pictures and tables for a complete report. Use these in the Results and Discussion sections.

### A) Core figures (must include)

1. Dataset shift summary (table or bar chart)
  - Show train vs test anomaly ratio and attack-type counts.
  - Purpose: motivate why a single detector can fail under shift.

2. Feature separability visualization (PCA and t-SNE)
  - Two scatter plots with normal vs attack coloring.
  - Purpose: show partial separability and overlap.

3. AE training curve (loss vs epoch)
  - Show train and validation loss for the autoencoder.
  - Purpose: confirm stable training and no overfitting.

4. ROC curves (AE, latent LR, gated OR)
  - Single plot with three ROC curves.
  - Purpose: compare overall ranking performance.

5. PR curves (AE, latent LR, gated OR)
  - Single plot with three PR curves.
  - Purpose: compare precision/recall trade-offs under class imbalance.

6. Confusion matrices (AE vs gated OR)
  - Side-by-side heatmaps.
  - Purpose: show concrete error changes in TP/FP/FN.

7. Attack-type recall comparison
  - Bar chart for top attack types (AE vs gated OR).
  - Purpose: show where the proposed method improves coverage.

8. Unified leaderboard (ML vs DL vs production)
  - Bar chart of F1 or summary table across all detectors.
  - Purpose: one-shot comparison across approaches.

### B) Proposed method validation (strongly recommended)

9. Ablation: F1 vs latent threshold quantile
  - Plot OR vs AND rules across quantiles.
  - Purpose: show calibration-driven threshold selection.

10. Bootstrap distribution of F1 delta
   - Histogram of (gated F1 - AE F1).
   - Purpose: show statistical confidence of improvement.

11. Seen vs unseen attack recall table
   - Small table with seen/unseen recall for AE and gated.
   - Purpose: show generalization across attack families.

12. Multi-seed stability table
   - Per-seed metrics + mean/std summary.
   - Purpose: show results are not single-run luck.

13. Paired significance table
   - t-test and Wilcoxon p-values for F1 and recall metrics.
   - Purpose: provide statistical significance evidence.

### C) Production pipeline evidence (optional but valuable)

14. Production tuned vs conformal comparison
   - Table of metrics for tuned and conformal thresholds.
   - Purpose: show deployment trade-offs (precision vs recall).

15. Risk tier distribution (production)
   - Bar chart of predicted risk tiers.
   - Purpose: show operational interpretability.

### D) ML baseline evidence (optional but useful)

16. ML model comparison bar chart
   - F1 or PR-AUC for Isolation Forest, LOF, One-Class SVM.
   - Purpose: justify the chosen ML baseline.

17. ML vs DL agreement table
   - Agreement and disagreement counts for ML vs AE and ML vs latent.
   - Purpose: show model diversity and complementary behavior.

## 16) Suggested figure file names

Use consistent naming so your report is easy to organize:

- figure_01_dataset_shift.png
- figure_02_pca_projection.png
- figure_03_tsne_projection.png
- figure_04_ae_training_curve.png
- figure_05_roc_comparison.png
- figure_06_pr_comparison.png
- figure_07_confusion_ae_vs_gated.png
- figure_08_attack_type_recall.png
- figure_09_ablation_quantiles.png
- figure_10_bootstrap_f1_delta.png
- figure_11_seen_unseen_recall.png
- figure_12_multi_seed_summary.png
- figure_13_paired_significance.png
- figure_14_leaderboard.png
- figure_15_prod_risk_tiers.png

If you want, I can export all plots directly from the notebooks into an output folder and name them using the list above.
