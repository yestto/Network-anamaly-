# Project Story: From Previous Baselines to the Proposed Solution

## Opening: The Original Goal

We started with a straightforward goal: detect network anomalies on NSL-KDD using a reliable and reproducible pipeline. The first version focused on two families of methods:

1. Classical ML baselines that were fast and easy to deploy.
2. A DL autoencoder that learned normal traffic and flagged deviations.

These baselines provided a strong starting point, but the dataset itself revealed a deeper challenge: the test split contains a higher anomaly ratio and more attack types than the training split. That shift means a single detector can be accurate overall but still miss specific attack families.

## The Previous Approach (Baseline Phase)

### ML Baselines (Previous)
The first phase used unsupervised ML detectors trained on normal traffic:
- Isolation Forest
- Local Outlier Factor (novelty mode)
- One-Class SVM

These models were good at precision and had stable, fast training. However, they typically traded recall for precision under distribution shift. In other words, they produced fewer false positives but could miss many attacks.

### ML Solution Details (Baseline Implementation)

The ML pipeline is a complete, reproducible baseline with the following steps:

1. Preprocessing
	- Categorical features encoded with an ordinal encoder.
	- Numeric features imputed (median) and scaled.
	- Train/test columns aligned for schema stability.

2. Models
	- Isolation Forest (best overall in this run)
	- Local Outlier Factor (novelty)
	- One-Class SVM

3. Evaluation
	- Accuracy, Precision, Recall, F1
	- ROC-AUC and PR-AUC
	- Confusion matrices
	- Threshold sensitivity analysis
	- Attack-type recall analysis

4. Example ML results (current run snapshot)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.8049 | 0.9718 | 0.6769 | 0.7980 | 0.9371 | 0.9527 |
| Local Outlier Factor | 0.7661 | 0.8246 | 0.7483 | 0.7846 | 0.8547 | 0.8228 |
| One-Class SVM | 0.7838 | 0.9203 | 0.6790 | 0.7814 | 0.8670 | 0.8765 |

5. ML artifacts produced
	- checkpoints/ml_models_major/
	  - isolation_forest.joblib
	  - local_outlier_factor.joblib
	  - oneclass_svm.joblib
	  - standard_scaler.joblib
	  - ordinal_encoder.joblib
	  - feature_names.joblib
	  - categorical_columns.joblib
	  - ml_metadata.json

6. ML inference output
	- ml_test_predictions.csv
	  - pred_label
	  - anomaly_score

This ML baseline serves as the precision-strong reference point that the DL methods and the proposed gated ensemble are compared against.

### DL Baseline (Previous)
The DL baseline was a reconstruction-only autoencoder:
- Train on normal traffic.
- Compute reconstruction error on test data.
- Flag anomalies above a threshold.

This improved recall and F1 compared to ML, but it still relied on a single signal. Some attacks reconstruct well (low error), which makes them invisible to a reconstruction-only detector.

**Limitation of previous approach:**
A single detector is not enough under real-world shift. Attacks can be abnormal in one representation but not another.

## The Turning Point: Why We Needed a New Method

When we analyzed errors by attack type, a pattern emerged:
- The autoencoder caught some families very well.
- The latent classifier (trained on AE embeddings) caught different families.

This means each detector covers different blind spots. The natural next step was not to replace one detector, but to combine them in a principled way.

## The Proposed Solution (Now)

### Calibrated Two-Stage Gated Ensemble
We introduced a new DL method that combines two detectors:

1. Stage 1: Autoencoder reconstruction threshold
2. Stage 2: Latent Logistic Regression classifier
3. Calibration: latent threshold set from a normal-validation split
4. Decision rule: gated OR (flag anomaly if either stage triggers)

This method is calibrated to avoid leakage and uses only validation data for threshold selection. The result is a detector that keeps precision strong while improving recall and F1.

**Why it is better than the previous method:**
- The baseline AE uses only reconstruction error.
- The new method adds a latent probability signal.
- Attacks missed by reconstruction can still be captured by latent risk.
- Seen-attack recall improves, and unseen-attack recall is maintained.

## The Broader System (Production-Grade Pipeline)

Beyond the proposed method, a production-ready pipeline was also built:

- Denoising autoencoder reconstruction
- Latent Logistic Regression
- Latent Isolation Forest
- Isotonic calibration for all components
- Stacked meta-classifier
- Tuned and conformal thresholds
- Risk tier outputs for operations

This pipeline turns the research method into an operational system with calibration, monitoring outputs, and stable inference behavior.

## Evidence and Validation

To avoid relying on a single lucky run, the evaluation includes:
- Multi-seed stability checks
- Paired significance tests (t-test and Wilcoxon)
- Attack-type recall breakdowns
- Optional external dataset validation

This creates a defensible claim: the new method consistently improves recall and F1 compared to a single-detector baseline under NSL-KDD shift.

## Final Outcome

**Before (previous):**
- ML baselines with strong precision but lower recall
- AE reconstruction baseline with better balance, but still single-signal

**Now (proposed):**
- A calibrated two-stage gated ensemble that combines reconstruction and latent signals
- Improved recall and F1 with minimal added complexity
- A production-grade hybrid pipeline for deployment and monitoring

This progression shows a clear research story: move from single-detector baselines to a calibrated, ensemble-based solution that addresses real-world distribution shift and delivers stronger detection coverage.
