# Network Anomaly Detection Modularization Project

## What This Is
Python-based network anomaly detection system modularized into ML and DL pipelines for scalable experimentation. Works on CSV network telemetry with preprocessing, representation learning (autoencoder), and downstream classification. The goal is a unified, configuration-driven framework that makes it easy to add, compare, and maintain multiple anomaly detection algorithms.

## Core Value
Provide a single, consistent pipeline that supports multiple ML/DL algorithms and evaluates them with unified metrics and configuration.

## Requirements

### Validated
(None yet — ship to validate)

### Active
- [ ] Modular ML pipeline: Isolation Forest, LOF, One-Class SVM
- [ ] Modular DL pipeline: Autoencoder, VAE, LSTM
- [ ] Unified training, prediction, and evaluation interfaces
- [ ] YAML-based configuration and reproducible experiments
- [ ] Plugin-based algorithm registration
- [ ] Unified evaluation and comparison tools
- [ ] Optional CLI for experiment execution

### Out of Scope
- Real-time streaming anomaly detection — future phase
- Distributed training/processing — future phase
- Web dashboards or REST API — out of scope for current work
- AutoML capabilities — future phase
- Federated learning — future phase

## Context
- Code anchors: `main.py`, `ml_models/autoencoder.py`, `util/data_processing.py`, `README.md`
- Tech stack: Python, Keras/TensorFlow, scikit-learn, pandas
- Planning artifacts: `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`
- Current code implements autoencoder training and a downstream logistic regression classifier for evaluation

## Constraints
- Tech: Use Keras/TensorFlow and scikit-learn for algorithm implementations — aligns with the existing stack
- Data: CSV inputs with label normalization and categorical encoding
- Performance: Keep training/inference practical for large CSVs; memory-aware batching
- Compatibility: Maintain simple CLI usage via `main.py` until unified CLI is introduced

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Separate ML and DL pipelines with shared interfaces | Clear modularity and extensibility | — Pending |
| Configuration-driven experiments (YAML) | Reproducibility and easy algorithm switching | — Pending |
| Plugin system for algorithm registration | Low-friction additions and maintenance | — Pending |
| Unified evaluation metrics and comparison | Consistent cross-approach assessment | — Pending |

---
*Last updated: 2026-02-15 after project initialization and modularization planning*
