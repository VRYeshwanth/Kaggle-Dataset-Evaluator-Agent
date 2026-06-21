"""
tools.py

MCP tools for Kaggle Dataset Evaluator Agent.
"""

from pathlib import Path

from src.analysis import (
    load_dataset,
    run_full_analysis,
)


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _load_analysis(dataset_path: str) -> dict:
    dataset_path = Path(dataset_path)

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    dataset_path = dataset_path.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = load_dataset(str(dataset_path))

    return run_full_analysis(df)


def analyze_dataset(dataset_path: str) -> dict:
    dataset_path = Path(dataset_path)

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    dataset_path = dataset_path.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = load_dataset(str(dataset_path))

    return run_full_analysis(df)


def get_dataset_score(
    dataset_path: str,
) -> dict:
    """
    Return dataset score only.
    """

    analysis = _load_analysis(
        dataset_path
    )

    return analysis["dataset_score"]


def get_project_recommendations(
    dataset_path: str,
) -> dict:
    """
    Return project recommendations.
    """

    analysis = _load_analysis(
        dataset_path
    )

    return {
        "project_recommendations":
        analysis[
            "project_recommendations"
        ]
    }


def generate_dataset_report(
    dataset_path: str,
) -> dict:
    """
    Return human-readable report.
    """

    analysis = _load_analysis(
        dataset_path
    )

    return {
        "report":
        analysis["report"]
    }