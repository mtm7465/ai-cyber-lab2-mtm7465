"""Data loading placeholders for Lab 2."""

from pathlib import Path


def get_data_paths(project_root: Path) -> dict[str, Path]:
    """Return canonical raw and processed data paths."""
    return {
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
    }
