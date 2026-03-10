# main.py CLI Guide

This document explains exactly what `main.py` does, what checkpoints mean, which commands to run, and what files are generated.

## 1) What is `main.py`?

`main.py` is the command-line execution layer of the project.

Use cases:
- run training without opening notebooks
- run inference on new data files
- generate reproducible artifacts/checkpoints
- automate runs in scripts/servers/CI

Think of it as:
- notebooks = exploration/visualization
- `main.py` = reproducible pipeline runner

## 2) What does `main.py` do internally?

At a high level it provides 5 command groups:
- `download-data` → downloads NSL-KDD files
- `train-ml` → trains classical ML anomaly models
- `test-ml` → runs ML inference using saved artifacts
- `train-dl` → trains deep-learning autoencoder pipeline
- `test-dl` → runs DL inference using saved artifacts

It also keeps legacy compatibility:
- `--mode train`
- `--mode test`

## 3) What is a checkpoint?

A checkpoint is the saved state of your trained pipeline, including model files + preprocessing objects + metadata.

Why checkpoints are important:
- inference must use the same preprocessing used during training
- you can re-run prediction without retraining
- experiments are reproducible

In this project, checkpoints are folders under `checkpoints/`.

## 4) ML checkpoint contents

Default folder:
- `checkpoints/ml_models_major/`

Typical files:
- `isolation_forest.joblib`
- `local_outlier_factor.joblib`
- `oneclass_svm.joblib`
- `standard_scaler.joblib`
- `ordinal_encoder.joblib`
- `feature_names.joblib`
- `categorical_columns.joblib`
- `ml_metadata.json`

Meaning:
- `*.joblib` model files = trained ML detectors
- scaler/encoder/feature files = exact preprocessing pipeline
- `ml_metadata.json` = summary metrics, best model info, labels

## 5) DL checkpoint contents

Default folder:
- `checkpoints/autoencoder_major/`

Typical files:
- `autoencoder_major.keras`
- `latent_classifier.joblib`
- `standard_scaler.joblib`
- `ordinal_encoder.joblib`
- `feature_names.joblib`
- `categorical_columns.joblib`
- `dl_metadata.json`

Meaning:
- `.keras` file = trained autoencoder
- latent classifier = classifier on encoder latent vectors
- preprocessing files = same transform pipeline as training
- `dl_metadata.json` = threshold + metrics + label mapping

## 6) Required dataset format

`main.py` uses NSL-KDD structure.

Expected files:
- `KDDTrain+.txt`
- `KDDTest+.txt`

Expected columns include:
- feature columns + `label` (+ optional `difficulty`)

Binary target mapping used by pipeline:
- `normal -> 0`
- `any attack label -> 1`

## 7) Commands to run (recommended)

Run from project root.

### Step 1: Download data
```bash
python main.py download-data --data_dir ./data
```
Output:
- `data/KDDTrain+.txt`
- `data/KDDTest+.txt`

### Step 2: Train ML
```bash
python main.py train-ml --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/ml_models_major
```
Output:
- ML checkpoint folder with models + preprocessors + metadata

### Step 3: Train DL
```bash
python main.py train-dl --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major
```
Output:
- DL checkpoint folder with autoencoder + preprocessors + metadata

### Step 4: Test ML
```bash
python main.py test-ml --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/ml_models_major --output_csv ./ml_test_predictions.csv
```
Output CSV columns added:
- `pred_label`
- `anomaly_score`

### Step 5: Test DL
```bash
python main.py test-dl --input_path ./data/KDDTest+.txt --model_dir ./checkpoints/autoencoder_major --output_csv ./dl_test_predictions.csv
```
Output CSV columns added:
- `pred_ae`
- `pred_latent_clf`
- `reconstruction_error`
- `latent_clf_score`

## 8) Legacy commands (still supported)

```bash
python main.py --mode train --data_path ./data/KDDTrain+.txt --ckpt_path ./checkpoints --model_name autoencoder
python main.py --mode test --data_path ./data/KDDTest+.txt --ckpt_path ./checkpoints --model_name autoencoder
```

Use modern subcommands for clearer and more controllable runs.

## 9) Typical output files after full run

Generated prediction files:
- `ml_test_predictions.csv`
- `dl_test_predictions.csv`
- `ml_vs_dl_predictions.csv` (if merged for side-by-side comparison)

Generated metadata files:
- `checkpoints/ml_models_major/ml_metadata.json`
- `checkpoints/autoencoder_major/dl_metadata.json`

## 10) Quick troubleshooting

- Error: model file not found in `test-ml`
  - Ensure `train-ml` ran successfully first
  - Check `model_dir` path

- Error: dataset file not found
  - Run `download-data` first or provide correct `--train_path/--test_path`

- Prediction output looks wrong
  - Verify inference uses matching checkpoint folder from the same training run

## 11) In short

`main.py` is your production-style runner:
- download data
- train ML/DL
- save checkpoints
- run inference on new data
- generate reproducible outputs
