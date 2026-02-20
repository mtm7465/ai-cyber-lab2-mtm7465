from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Constants (can be overridden when calling the function)
DATA_PATH = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
LABEL_COL = "label"
TEXT_COL = "text"


def load_and_split(
    data_path: str = DATA_PATH,
    label_col: str = LABEL_COL,
    text_col: Optional[str] = TEXT_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """Load CSV data, do basic cleaning, and return a stratified train/test split.

    Track 1 note: PhiUSIIL uses engineered numeric URL features; `text_col` is optional.

    Behavior:
    - Loads `data_path` (CSV) into a DataFrame.
    - Drops rows with missing labels and strips string columns.
    - If `text_col` is provided and present, treats this as a text-classification
      problem and returns TF-IDF features for X.
    - Otherwise treats remaining numeric columns as features for tabular data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {data_path}")

    # Drop rows with missing label and strip string columns
    df = df.dropna(subset=[label_col])
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    if text_col and text_col in df.columns:
        df = df.dropna(subset=[text_col])
        texts = df[text_col].astype(str)
        labels = df[label_col].to_numpy()

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        return X_train, X_test, y_train, y_test

    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric feature columns found for tabular mode")

    df = df.dropna(subset=numeric_cols)

    X = df[numeric_cols].to_numpy()
    y = df[label_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
