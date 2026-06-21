"""
security.py

Dataset path validation utilities.
"""

from pathlib import Path


ALLOWED_EXTENSIONS = {".csv"}

MAX_FILE_SIZE_MB = 100


def validate_dataset_path(dataset_path: str) -> str:
    """
    Validate dataset path before processing.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise ValueError(
            f"Dataset does not exist: {dataset_path}"
        )

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Only CSV files are allowed."
        )

    file_size_mb = (
        path.stat().st_size
        / (1024 * 1024)
    )

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Dataset exceeds {MAX_FILE_SIZE_MB} MB limit."
        )

    return str(path.resolve())