"""
tools.py

MCP tools for Kaggle Dataset Evaluator Agent.
"""

from pathlib import Path
from functools import lru_cache
from agent.security import validate_dataset_path

from src.analysis import (
    load_dataset,
    run_full_analysis,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def _resolve_dataset_path(
    dataset_path: str,
) -> str:
    """
    Resolve relative paths and apply
    security validation.
    """

    path = Path(dataset_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    return validate_dataset_path(
        str(path)
    )

@lru_cache(maxsize=32)
def _load_analysis(
    dataset_path: str,
) -> dict:
    """
    Resolve, validate, load and analyze
    the dataset.
    """

    dataset_path = (
        _resolve_dataset_path(
            dataset_path
        )
    )

    df = load_dataset(dataset_path)

    return run_full_analysis(df)

def analyze_dataset(
    dataset_path: str,
) -> dict:
    """
    Return complete dataset analysis.
    """

    try:

        print(
            f"[MCP] analyze_dataset -> "
            f"{dataset_path}"
        )

        return _load_analysis(
            dataset_path
        )

    except Exception as e:

        return {
            "error": str(e)
        }


def get_dataset_score(
    dataset_path: str,
) -> dict:
    """
    Return dataset score.
    """

    try:

        print(
            f"[MCP] dataset_score -> "
            f"{dataset_path}"
        )

        analysis = _load_analysis(
            dataset_path
        )

        return analysis[
            "dataset_score"
        ]

    except Exception as e:

        return {
            "error": str(e)
        }


def get_project_recommendations(
    dataset_path: str,
) -> dict:
    """
    Return project recommendations.
    """

    try:

        print(
            "[MCP] "
            "project_recommendations -> "
            f"{dataset_path}"
        )

        analysis = _load_analysis(
            dataset_path
        )

        return {
            "project_recommendations":
            analysis[
                "project_recommendations"
            ]
        }

    except Exception as e:

        return {
            "error": str(e)
        }


def generate_dataset_report(
    dataset_path: str,
) -> dict:
    """
    Return human-readable report.
    """

    try:

        print(
            "[MCP] "
            "generate_report -> "
            f"{dataset_path}"
        )

        analysis = _load_analysis(
            dataset_path
        )

        return {
            "report":
            analysis["report"]
        }

    except Exception as e:

        return {
            "error": str(e)
        }