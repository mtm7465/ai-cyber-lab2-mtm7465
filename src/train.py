"""Training entry point placeholder."""

from pathlib import Path

from .data import get_data_paths
from .utils import log


def main() -> None:
    """Run placeholder training workflow."""
    project_root = Path(__file__).resolve().parents[1]
    paths = get_data_paths(project_root)
    log("Starting placeholder training run")
    log(f"Using raw data directory: {paths['raw']}")
    log(f"Using processed data directory: {paths['processed']}")
    log("Training complete (placeholder).")


if __name__ == "__main__":
    main()
