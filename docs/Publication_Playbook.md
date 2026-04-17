# Publication Playbook (DL Gated Ensemble)

This guide defines a reproducible protocol for publication-grade results using `main.py eval-publication`.

## 1) Goal

Demonstrate that the gated DL method is robust across random seeds and statistically better than the single-detector baseline on core metrics, while reporting trade-offs clearly.

## 2) Baseline and Proposed Method

- Baseline: reconstruction-only autoencoder detector (`AE`)
- Proposed: calibrated gated OR detector (`Gated-OR`)

Both are evaluated on the same test split per seed.

## 3) Quick Dry-Run (sanity check)

Run a fast validation first:

```bash
python main.py eval-publication --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major_project_gated_pub_quick --seeds 7,21 --sample_fraction 0.15 --epochs 2 --target_fprs 0.01,0.05
```

Use this to verify pipeline wiring and output files.

## 4) Final Paper Run (recommended)

Run full reproducible evaluation:

```bash
python main.py eval-publication --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major_project_gated --seeds 7,21,42,84,126 --sample_fraction 1.0 --epochs 30 --target_fprs 0.01,0.05,0.10
```

## 5) Optional External Validation

If you have an external dataset CSV:

```bash
python main.py eval-publication --train_path ./data/KDDTrain+.txt --test_path ./data/KDDTest+.txt --output_dir ./checkpoints/autoencoder_major_project_gated --seeds 7,21,42,84,126 --sample_fraction 1.0 --epochs 30 --external_path ./data/EXTERNAL_TEST.csv --external_label_column label --external_mapping_json ./docs/external_feature_mapping_template.json --save_external_predictions
```

Edit the mapping JSON for your external schema.

## 6) Output Artifacts

- `publication_seed_metrics.csv`
- `publication_paired_significance.csv`
- `publication_attack_type_by_seed.csv`
- `publication_attack_type_summary.csv`
- `publication_report.json`
- `publication_external_metrics_by_seed.csv` (only when external data is provided)

## 7) Reporting Template

Minimum table set for paper:

1. Seed-wise and mean/std metrics: AE vs Gated-OR (`f1`, `recall`, `precision`, `seen_recall`, `unseen_recall`, `normal_fpr`)
2. Paired significance table (`ttest_pvalue`, `wilcoxon_pvalue`)
3. Attack-type delta recall table (top gains + top remaining failures)
4. Optional external dataset table (same metrics + deltas)

## 8) Claim Discipline

Safe claim:

"The calibrated gated ensemble consistently improves overall F1 and recall over the AE baseline under NSL-KDD shift, with explicit false-positive-rate trade-offs."

Do not claim universal zero-day improvement unless external validation confirms it.
