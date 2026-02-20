"""Evaluation entry point placeholder."""

from pathlib import Path

from .utils import log


def main() -> None:
    """Run placeholder evaluation workflow."""
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    log("Starting placeholder evaluation run")
    log(f"Results directory: {results_dir}")
    log("Evaluation complete (placeholder).")


if __name__ == "__main__":
    main()
