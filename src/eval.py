"""Evaluate a trained model or train deterministically and evaluate.

Run as: python -m src.eval
"""
from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .data import load_and_split, DATA_PATH, LABEL_COL, TEXT_COL

MODEL_PATH = "results/model.joblib"
METRICS_PATH = "results/metrics.json"
CONF_MATRIX_PATH = "results/confusion_matrix.png"
RANDOM_SEED = 42


def ensure_results_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    ensure_results_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def evaluate(
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
    conf_path: str = CONF_MATRIX_PATH,
    data_path: str = DATA_PATH,
    label_col: str = LABEL_COL,
    text_col: Optional[str] = TEXT_COL,
):
    # Load data split deterministically
    X_train, X_test, y_train, y_test = load_and_split(
        data_path=data_path, label_col=label_col, text_col=text_col, random_state=RANDOM_SEED
    )

    model_loaded = None
    if os.path.exists(model_path):
        model_loaded = joblib.load(model_path)
        model = model_loaded
    else:
        # Train deterministic model inline (LogisticRegression)
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        X_train_fit = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        model.fit(X_train_fit, y_train)
        ensure_results_dir(model_path)
        joblib.dump(model, model_path)

    # Prepare test features for prediction
    X_test_eval = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    y_pred = model.predict(X_test_eval)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    labels = np.unique(np.concatenate([y_test, y_pred])).tolist()
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "n_test": int(len(y_test)),
    }

    ensure_results_dir(metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    plot_confusion_matrix(cm, labels=[str(l) for l in labels], out_path=conf_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {conf_path}")
    return metrics


def main():
    evaluate()


if __name__ == "__main__":
    main()
