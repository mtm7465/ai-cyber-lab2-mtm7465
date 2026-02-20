from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Constants (can be overridden when calling the function)
DATA_PATH = "data/raw/dataset.csv"
LABEL_COL = "label"
TEXT_COL = "text"


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from object (string) columns in-place and return df."""
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_and_split(
    data_path: str = DATA_PATH,
    label_col: str = LABEL_COL,
    text_col: Optional[str] = TEXT_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """Load CSV data, do basic cleaning, and return a stratified train/test split.

    Behavior:
    - Loads `data_path` (CSV) into a DataFrame.
    - Drops rows with missing labels and strips string columns.
    - If `text_col` is provided and present in the DataFrame, treats this as
      a text-classification problem: returns TF-IDF features for X.
    - Otherwise treats remaining numeric columns as features for tabular data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(data_path)

    # Ensure label exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {data_path}")

    # Drop rows with missing label
    df = df.dropna(subset=[label_col])

    # Basic cleaning: strip string columns
    df = _strip_string_columns(df)

    # If text_col is provided and exists, run TF-IDF on that column
    if text_col and text_col in df.columns:
        # Drop rows missing the text
        df = df.dropna(subset=[text_col])

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].values

        # Stratified split on labels
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        return X_train, X_test, y_train, y_test

    # Otherwise assume numeric/tabular features (exclude the label column)
    # Try to coerce non-numeric columns (except label) to numeric where possible
    feature_cols = [c for c in df.columns if c != label_col]

    # Attempt to convert candidate feature columns to numeric when possible
    for c in feature_cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Select numeric columns as features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    if not numeric_cols:
        raise ValueError("No numeric feature columns found for tabular mode")

    # Drop rows with missing features
    df = df.dropna(subset=numeric_cols + [label_col])

    X = df[numeric_cols].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Constants (can be overridden when calling the function)
DATA_PATH = "data/raw/dataset.csv"
LABEL_COL = "label"
TEXT_COL = "text"


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from object (string) columns in-place and return df."""
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_and_split(
    data_path: str = DATA_PATH,
    label_col: str = LABEL_COL,
    text_col: Optional[str] = TEXT_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """Load CSV data, do basic cleaning, and return a stratified train/test split.

    Behavior:
    - Loads `data_path` (CSV) into a DataFrame.
    - Drops rows with missing labels and strips string columns.
    - If `text_col` is provided and present in the DataFrame, treats this as
      a text-classification problem: returns TF-IDF features for X.
    - Otherwise treats remaining numeric columns as features for tabular data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(data_path)

    # Ensure label exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {data_path}")

    # Drop rows with missing label
    df = df.dropna(subset=[label_col])

    # Basic cleaning: strip string columns
    df = _strip_string_columns(df)

    # If text_col is provided and exists, run TF-IDF on that column
    if text_col and text_col in df.columns:
        # Drop rows missing the text
        df = df.dropna(subset=[text_col])

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].values

        # Stratified split on labels
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        return X_train, X_test, y_train, y_test

    # Otherwise assume numeric/tabular features (exclude the label column)
    # Try to coerce non-numeric columns (except label) to numeric where possible
    feature_cols = [c for c in df.columns if c != label_col]

    # Attempt to convert candidate feature columns to numeric when possible
    for c in feature_cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Select numeric columns as features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    if not numeric_cols:
        raise ValueError("No numeric feature columns found for tabular mode")

    # Drop rows with missing features
    df = df.dropna(subset=numeric_cols + [label_col])

    X = df[numeric_cols].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
"""Data loading placeholders for Lab 2."""

from pathlib import Path


def get_data_paths(project_root: Path) -> dict[str, Path]:
    """Return canonical raw and processed data paths."""
    return {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
    }
