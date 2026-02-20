# AI Cyber Lab 2 - Track 1 Phishing Detection (Baseline ML Pipeline)

## Project Description
This project implements a minimal, reproducible baseline for **Track 1: Phishing Detection** (binary URL classification) using engineered URL features in a tabular ML pipeline. It is designed for reproducibility and clean experiment workflow rather than model complexity.

## Dataset Source and Features
- **Source:** PhiUSIIL Phishing URL Dataset at https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset.
- **Label column:** `label` (phishing vs. benign).
- **Feature columns:** Engineered numeric URL features (all numeric columns except `label`).

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
Metrics from the latest evaluation run:
- `accuracy`: 0.9997243368
- `precision_macro`: 0.9997216279
- `recall_macro`: 0.9997154036
- `f1_macro`: 0.9997185152
- `n_test`: 47159

These results are written to `results/metrics.json`, and the confusion matrix is saved to `results/confusion_matrix.png` by `python -m src.eval`.

**Caution:** Very high performance may reflect curated engineered features or dataset-specific patterns and may not fully represent real-world phishing detection difficulty.

## Ethics and Safety Considerations
- False positives can disrupt legitimate traffic; false negatives can allow malicious URLs through.
- Model performance can degrade under distribution shift and attacker adaptation; monitor and retrain responsibly.
- Dataset bias may skew results; validate on diverse sources before deployment.
- Outputs can be misused to optimize phishing techniques; restrict access to models and features.
- This is a baseline model and should not be used for production security decisions without rigorous testing.

## Reproducibility
All splits and models use a fixed random seed (`42`) for deterministic results.
