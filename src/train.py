"""Train a baseline classifier and optionally save the trained model.

Run as: python -m src.train
"""
from __future__ import annotations

import argparse
import os
from pprint import pformat

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

from .data import load_and_split, DATA_PATH, LABEL_COL, TEXT_COL


# Defaults / constants
DEFAULT_MODEL = "logistic"  # options: logistic, randomforest, linearsvc
MODEL_PATH = "results/model.joblib"
RANDOM_SEED = 42


def get_model(name: str, random_state: int = RANDOM_SEED):
    name = name.lower()
    if name == "logistic":
        return LogisticRegression(random_state=random_state, max_iter=1000)
    if name == "randomforest":
        return RandomForestClassifier(random_state=random_state)
    if name == "linearsvc":
        return LinearSVC(random_state=random_state, max_iter=10000)
    raise ValueError(f"Unknown model '{name}'")


def ensure_results_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def train_and_report(
    model_name: str = DEFAULT_MODEL,
    data_path: str = DATA_PATH,
    label_col: str = LABEL_COL,
    text_col: str | None = TEXT_COL,
    save_path: str | None = MODEL_PATH,
):
    X_train, X_test, y_train, y_test = load_and_split(
        data_path=data_path, label_col=label_col, text_col=text_col, random_state=RANDOM_SEED
    )

    model = get_model(model_name, random_state=RANDOM_SEED)

    # If features are sparse (e.g., TF-IDF) most linear models accept them.
    # RandomForest requires dense arrays.
    X_train_fit = X_train
    X_test_eval = X_test
    if hasattr(X_train, "toarray") and model_name == "randomforest":
        X_train_fit = X_train.toarray()
        X_test_eval = X_test.toarray()

    model.fit(X_train_fit, y_train)

    y_pred = model.predict(X_test_eval)

    acc = accuracy_score(y_test, y_pred)

    summary = {
        "model": model_name,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": float(acc),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    print("Training summary:\n", pformat(summary))

    if save_path:
        ensure_results_dir(save_path)
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")

    return model, summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a baseline classifier")
    p.add_argument("--model", choices=["logistic", "randomforest", "linearsvc"], default=DEFAULT_MODEL)
    p.add_argument("--data", default=DATA_PATH, help="Path to CSV dataset")
    p.add_argument("--label", default=LABEL_COL, help="Label column name")
    p.add_argument("--text-col", default=TEXT_COL, help="Text column for TF-IDF (optional)")
    p.add_argument("--no-save", dest="save", action="store_false", help="Do not save trained model")
    p.add_argument("--out", default=MODEL_PATH, help="Path to save trained model")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    save_path = args.out if args.save else None
    train_and_report(
        model_name=args.model,
        data_path=args.data,
        label_col=args.label,
        text_col=args.text_col,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
