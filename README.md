# AI Cyber Lab 2 - Baseline ML Pipeline

## Project Description
This project implements a minimal, reproducible baseline for a cybersecurity classification task (Lab 2 track). It supports both text classification (TF-IDF + linear models) and numeric tabular classification with standard scikit-learn models.

## Dataset Source and Features
- **Source:** Course-provided Lab 2 dataset located at `data/raw/dataset.csv`.
- **Label column:** `label` (configurable).
- **Text feature column (optional):** `text` (used for TF-IDF if present).
- **Tabular features:** All numeric columns except `label` when `text` is not present.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
python -m src.train
```
Options:
- `--model` in `{logistic, randomforest, linearsvc}`
- `--data`, `--label`, `--text-col`
- `--no-save` and `--out` for model persistence

## Evaluation
```bash
python -m src.eval
```
This writes:
- `results/metrics.json`
- `results/confusion_matrix.png`

## Baseline Results
`results/metrics.json` not found yet. Run evaluation to generate baseline metrics.

## Ethics and Safety Considerations
- This is a baseline model and should not be used for production security decisions.
- Class imbalance can hide poor performance on minority classes; use macro-averaged metrics.
- Sensitive or personal data should be handled according to organizational policies.
- Model predictions can be biased by data collection artifacts; validate across sources.
- Keep training data and outputs secured to avoid leaking security-relevant signals.

## Reproducibility
All splits and models use a fixed random seed (`42`) for deterministic results.
