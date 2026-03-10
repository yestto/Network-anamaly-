# ML vs DL Comparison for Network Anomaly Detection

For a complete implementation log (what was built, commands executed, outputs generated, and `main.py` changes), see:
- `docs/Implementation_Summary.md`

This document compares the two project notebooks:
- `notebooks/ML_Anomaly_Detection.ipynb`
- `notebooks/DL_Anomaly_Detection.ipynb`

Both pipelines use **NSL-KDD** and follow the same preprocessing/evaluation philosophy (binary target: `normal=0`, `attack=1`).

## 1) Approaches

### Classical ML Notebook
Unsupervised anomaly methods trained on normal traffic:
- Isolation Forest
- Local Outlier Factor (novelty mode)
- One-Class SVM

Then evaluated on full test data with:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Confusion matrix
- Threshold sensitivity
- Attack-type recall analysis

### Deep Learning Notebook
Deep anomaly pipeline using:
- Autoencoder trained on normal traffic
- Reconstruction error scoring
- Latent representation extraction
- Downstream classifier and threshold calibration

Then evaluated with the same metric family and richer latent/representation visuals.

## 2) ML Results (current run)

From the executed `ML_Anomaly_Detection.ipynb`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.8049 | 0.9718 | 0.6769 | 0.7980 | 0.9371 | 0.9527 |
| Local Outlier Factor | 0.7661 | 0.8246 | 0.7483 | 0.7846 | 0.8547 | 0.8228 |
| One-Class SVM | 0.7838 | 0.9203 | 0.6790 | 0.7814 | 0.8670 | 0.8765 |

Best overall in this run: **Isolation Forest** (strongest ROC-AUC/PR-AUC and F1).

## 3) DL vs ML: Practical Trade-offs

### When ML is stronger
- Faster training and easier to tune for baseline deployments.
- Lower compute/memory cost for CPU-only environments.
- Very fast iteration for feature engineering and threshold calibration.

### When DL is stronger
- Better at learning compact nonlinear representations in complex traffic patterns.
- Useful when feature interactions are highly nonlinear.
- Latent space can support richer downstream analysis and hybrid modeling.

### Typical operational differences
- ML: simpler pipeline, easier explainability, lower maintenance overhead.
- DL: higher complexity, usually higher compute cost, but often better flexibility/representation power.

## 4) Recommendation for this project

1. Use **Isolation Forest** as the production baseline from ML notebook artifacts.
2. Keep the DL autoencoder pipeline as an advanced path for improved representation learning.
3. Monitor both with the same threshold policy and attack-type recall dashboard.
4. Recalibrate thresholds periodically as traffic distributions drift.

## 5) Artifacts and deployment readiness

### ML artifacts
Saved in `checkpoints/ml_models_major/`:
- `standard_scaler.joblib`
- `ordinal_encoder.joblib`
- `categorical_columns.joblib`
- `feature_names.joblib`
- `isolation_forest.joblib`
- `local_outlier_factor.joblib`
- `oneclass_svm.joblib`
- `ml_metadata.json`

### DL artifacts
Saved in `checkpoints/autoencoder_major/` (and related DL checkpoint path from notebook):
- trained autoencoder model
- scaler/encoders/metadata used by DL pipeline

## 6) How to reproduce and compare quickly

1. Run `notebooks/ML_Anomaly_Detection.ipynb` end-to-end.
2. Run `notebooks/DL_Anomaly_Detection.ipynb` end-to-end.
3. Compare:
   - ROC-AUC and PR-AUC
   - F1 at calibrated threshold
   - attack-type recall spread
   - runtime and model size

This gives both a scientific comparison (detection quality) and an engineering comparison (cost and maintainability).

## 7) CLI commands to run (end-to-end)

Run from project root:

```bash
# 1) Download NSL-KDD files
python main.py download-data --data_dir ./data

# 2) Train ML pipeline
python main.py train-ml --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/ml_models_major

# 3) Train DL pipeline
python main.py train-dl --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major

# 4) Run ML inference
python main.py test-ml --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/ml_models_major --output_csv ./ml_test_predictions.csv

# 5) Run DL inference
python main.py test-dl --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/autoencoder_major --output_csv ./dl_test_predictions.csv
```

## 8) What outputs are generated

### Training outputs

- ML checkpoint folder: `checkpoints/ml_models_major/`
   - `isolation_forest.joblib`
   - `local_outlier_factor.joblib`
   - `oneclass_svm.joblib`
   - `standard_scaler.joblib`
   - `ordinal_encoder.joblib`
   - `feature_names.joblib`
   - `categorical_columns.joblib`
   - `ml_metadata.json`

- DL checkpoint folder: `checkpoints/autoencoder_major/`
   - `autoencoder_major.keras`
   - `latent_classifier.joblib`
   - `standard_scaler.joblib`
   - `ordinal_encoder.joblib`
   - `feature_names.joblib`
   - `categorical_columns.joblib`
   - `dl_metadata.json`

### Inference outputs

- `ml_test_predictions.csv`
   - includes original NSL-KDD columns + `pred_label`, `anomaly_score`

- `dl_test_predictions.csv`
   - includes original NSL-KDD columns +
      - `pred_ae`
      - `pred_latent_clf`
      - `reconstruction_error`
      - `latent_clf_score`

## 9) Side-by-side comparison output

Created comparison file:

- `ml_vs_dl_predictions.csv`

It contains:
- original NSL-KDD columns
- ML columns: `ml_pred_label`, `ml_anomaly_score`
- DL columns: `dl_pred_ae`, `dl_pred_latent_clf`, `dl_reconstruction_error`, `dl_latent_clf_score`
- agreement flags:
   - `ml_vs_ae_agree`
   - `ml_vs_latent_agree`

## 10) Current run summary (from generated files)

Dataset size compared:
- Rows: `22544`

Agreement between ML (Isolation Forest) and DL outputs:
- ML vs DL-AE agreement: `87.95%` (`2716` disagreements)
- ML vs DL-latent-classifier agreement: `89.67%` (`2329` disagreements)

Current DL metrics (`checkpoints/autoencoder_major/dl_metadata.json`):
- Autoencoder detector:
   - Accuracy: `0.8620`
   - Precision: `0.9366`
   - Recall: `0.8125`
   - F1: `0.8701`
   - ROC-AUC: `0.9563`
   - PR-AUC: `0.9475`
- Latent classifier:
   - Accuracy: `0.8059`
   - Precision: `0.9598`
   - Recall: `0.6879`
   - F1: `0.8014`
   - ROC-AUC: `0.9061`
   - PR-AUC: `0.9359`

Current ML metrics (`checkpoints/ml_models_major/ml_metadata.json`):
- Isolation Forest (best ML):
   - Accuracy: `0.8049`
   - Precision: `0.9718`
   - Recall: `0.6769`
   - F1: `0.7980`
   - ROC-AUC: `0.9371`
   - PR-AUC: `0.9527`

Interpretation of this run:
- DL autoencoder gives better recall and higher F1 than best ML baseline.
- ML Isolation Forest gives very high precision and strong PR-AUC.
- Both are useful: ML for precision-focused baseline and DL for stronger balanced detection.
