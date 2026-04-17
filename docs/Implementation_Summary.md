# Implementation Summary (ML, DL, ML-vs-DL, and main.py)

This document summarizes all major work completed in this project iteration, including notebook upgrades, CLI modernization, executed commands, generated outputs, and fixes.

## 1) What was upgraded

### A) Deep Learning notebook upgrade
File:
- `notebooks/DL_Anomaly_Detection.ipynb`

What was done:
- Upgraded to a major-project level pipeline with deeper EDA and richer visual analysis.
- Kept a real benchmark dataset flow (NSL-KDD) with robust loading and preprocessing.
- Added advanced evaluation:
  - ROC/PR curves
  - confusion matrices
  - threshold sensitivity analysis
  - attack-type recall analysis
- Improved artifact persistence and inference flow.
- Fixed Keras save behavior by using modern `.keras` model save format.

### B) Machine Learning notebook upgrade
File:
- `notebooks/ML_Anomaly_Detection.ipynb`

What was done:
- Upgraded to major-project quality to match DL depth.
- Added deep EDA and advanced diagnostics.
- Built complete classical anomaly benchmarking:
  - Isolation Forest
  - Local Outlier Factor (novelty mode)
  - One-Class SVM
- Added full evaluation dashboard:
  - ROC/PR comparison
  - confusion matrices
  - metric comparison bars
  - threshold calibration
  - attack-type recall analysis
- Added artifact save/load + inference utility for CSV prediction.
- Executed notebook cells and fixed runtime issues during validation.

### C) ML-vs-DL comparison deliverables
Files:
- `docs/ML_vs_DL_Comparison.md`
- `ml_test_predictions.csv`
- `dl_test_predictions.csv`
- `ml_vs_dl_predictions.csv`

What was done:
- Added documentation comparing ML and DL methods, trade-offs, metrics, and deployment recommendations.
- Generated side-by-side prediction dataset (`ml_vs_dl_predictions.csv`) including agreement flags.
- Added current-run agreement/disagreement statistics and practical interpretation.

### D) Production-grade hybrid pipeline (new)
Files:
- `main.py`
- `docs/Main_CLI_Guide.md`

What was done:
- Added two new CLI commands:
  - `train-prod`
  - `test-prod`
- Implemented a production hybrid ensemble architecture:
  - denoising autoencoder for reconstruction anomaly signal
  - latent-space Logistic Regression classifier
  - latent-space Isolation Forest detector
  - isotonic calibration for all component scores
  - stacked meta-classifier over calibrated component probabilities
  - tuned threshold (F1-optimized on holdout calibration subset)
  - conformal threshold (normal-holdout quantile based)
- Added production artifact contract:
  - model files
  - calibration models
  - threshold JSON
  - metadata JSON
  - monitoring baseline JSON (reference stats)
- Added production inference output schema including:
  - tuned and conformal predictions
  - combined anomaly probability
  - component probabilities/scores
  - dominant detector and risk tier

## 2) main.py modernization

File:
- `main.py`

Old state:
- Legacy train/test flow with older assumptions and less aligned behavior vs notebooks.

New state:
- Converted to notebook-aligned CLI with explicit subcommands:
  - `download-data`
  - `train-ml`
  - `test-ml`
  - `train-dl`
  - `test-dl`
  - `train-prod`
  - `test-prod`
- Added NSL-KDD download helper and standardized data handling.
- Added consistent preprocessing persistence for inference reproducibility:
  - scaler
  - ordinal encoder
  - feature names
  - categorical columns
- Added ML artifact metadata and DL artifact metadata.
- Kept legacy compatibility path:
  - `--mode train`
  - `--mode test`

### Important bug fix in main.py
Issue fixed:
- `test-ml` initially failed because metadata stored display model name (for example, `Isolation Forest`) while model filename used snake_case (`isolation_forest.joblib`).

Fix applied:
- Added robust model-name normalization + alias mapping + fallback file resolution in `run_test_ml`.

Result:
- `test-ml` now runs successfully with current metadata/artifact naming.

## 3) Commands that were run and validated

### Data preparation
```bash
python main.py download-data --data_dir ./data
```
Output:
- `data/KDDTrain+.txt`
- `data/KDDTest+.txt`

### DL training
```bash
python main.py train-dl --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major --epochs 5 --batch_size 256
```
Observed output summary:
- DL training executed successfully.
- Metrics printed for autoencoder and latent-classifier paths.
- Artifacts saved to `checkpoints/autoencoder_major`.

### DL inference
```bash
python main.py test-dl --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/autoencoder_major --output_csv ./dl_test_predictions.csv
```
Output:
- `dl_test_predictions.csv` created with:
  - `pred_ae`
  - `pred_latent_clf`
  - `reconstruction_error`
  - `latent_clf_score`

### Production hybrid training
```bash
python main.py train-prod --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/production_hybrid --epochs 2 --batch_size 512 --if_estimators 150
```
Observed output summary:
- training completed successfully and produced calibrated hybrid artifacts
- sample run metrics:
  - production tuned F1: 0.7599
  - production conformal F1: 0.8045
  - production ROC-AUC: 0.9379

### Production hybrid inference
```bash
python main.py test-prod --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/production_hybrid --output_csv ./prod_test_predictions.csv
```
Output:
- `prod_test_predictions.csv` created with production columns:
  - `pred_prod_tuned`
  - `pred_prod_conformal`
  - `prod_anomaly_probability`
  - `dominant_detector`
  - `risk_tier`

### ML inference
```bash
python main.py test-ml --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/ml_models_major --output_csv ./ml_test_predictions.csv
```
Output:
- `ml_test_predictions.csv` created with:
  - `pred_label`
  - `anomaly_score`

### Side-by-side merge generation
A merge command was executed to generate:
- `ml_vs_dl_predictions.csv`

This file includes both ML and DL predictions and agreement flags:
- `ml_vs_ae_agree`
- `ml_vs_latent_agree`

## 4) Artifacts generated

### ML artifacts
Folder:
- `checkpoints/ml_models_major`

Includes:
- `isolation_forest.joblib`
- `local_outlier_factor.joblib`
- `oneclass_svm.joblib`
- `standard_scaler.joblib`
- `ordinal_encoder.joblib`
- `feature_names.joblib`
- `categorical_columns.joblib`
- `ml_metadata.json`

### DL artifacts
Folder:
- `checkpoints/autoencoder_major`

Includes:
- `autoencoder_major.keras`
- `latent_classifier.joblib`
- `standard_scaler.joblib`
- `ordinal_encoder.joblib`
- `feature_names.joblib`
- `categorical_columns.joblib`
- `dl_metadata.json`

### Inference outputs
- `ml_test_predictions.csv`
- `dl_test_predictions.csv`
- `prod_test_predictions.csv`
- `ml_vs_dl_predictions.csv`

### Production artifacts
Folder:
- `checkpoints/production_hybrid`

Includes:
- `prod_autoencoder.keras`
- `prod_latent_logreg.joblib`
- `prod_latent_isolation_forest.joblib`
- `prod_meta_classifier.joblib`
- `prod_recon_isotonic.joblib`
- `prod_latent_logreg_isotonic.joblib`
- `prod_latent_if_isotonic.joblib`
- `prod_thresholds.json`
- `prod_metadata.json`
- `prod_monitor_baseline.json`

## 5) Current run snapshot (from generated metadata)

DL (autoencoder path) achieved stronger balanced detection in this run, while ML (Isolation Forest) remained a high-precision baseline.

Practical reading of results:
- Use ML Isolation Forest when precision-priority and simple deployment are most important.
- Use DL autoencoder path when stronger recall/F1 balance is needed.
- Keep both paths available and monitor drift with periodic threshold recalibration.

## 6) Where to read detailed comparison

Use:
- `docs/ML_vs_DL_Comparison.md`

That document contains:
- exact CLI workflow
- output descriptions
- ML-vs-DL agreement stats
- run-specific metric summaries
