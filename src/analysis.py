from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#----------------------------------------------------------------------
# JSON-safety helpers
#----------------------------------------------------------------------

def to_json_safe(value: Any) -> Any:
    """Recursively convert NumPy / pandas scalar types into native Python types.

    Guarantees that the returned object contains only ``float``, ``int``,
    ``bool``, ``str``, ``list``, ``dict``, or ``None`` -- safe for
    ``json.dumps`` and for crossing process/agent boundaries.

    Args:
        value: Any Python object, potentially containing NumPy/pandas
            scalars, arrays, or nested containers thereof.

    Returns:
        The same structure with all NumPy/pandas scalar types converted
        to native Python equivalents. NaN/NaT values are converted to
        ``None``.
    """
    if isinstance(value, dict):
        return {str(_to_json_safe_key(k)): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        as_float = float(value)
        return None if np.isnan(as_float) else as_float
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if value is pd.NaT:
        return None
    if isinstance(value, (np.str_,)):
        return str(value)
    return value


def _to_json_safe_key(key: Any) -> Any:
    """Convert a dict key into a JSON-safe (string-able) key."""
    if isinstance(key, (np.integer, np.floating, np.bool_)):
        return to_json_safe(key)
    return key


#----------------------------------------------------------------------
# Dtype classification helpers
#----------------------------------------------------------------------

def _column_dtype_kind(series: pd.Series) -> str:
    """Classify a pandas Series into a coarse dtype kind.

    Args:
        series: The column to classify.

    Returns:
        One of "boolean", "datetime", "numeric", "categorical", or "other".
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        return "categorical"
    if pd.api.types.is_string_dtype(series):
        return "categorical"
    return "other"


def _is_integer_like(series: pd.Series) -> bool:
    """Determine whether a numeric series only contains whole-number values.

    Handles both true integer dtypes and float columns that happen to
    store only integral values (e.g. ``1.0, 2.0, 3.0``).

    Args:
        series: A numeric pandas Series.

    Returns:
        True if every non-null value is a whole number.
    """
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        return bool(np.all(np.mod(non_null.to_numpy(dtype=float), 1) == 0))
    return False


#----------------------------------------------------------------------
# Identifier column detection
#----------------------------------------------------------------------

_ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|^index$|^idx$|^pk$|_pk$|"
    r"candidate_id|employee_id|customer_id|user_id|record_id|row_id|"
    r"^key$|_key$)",
    re.IGNORECASE,
)


def detect_identifier_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are most likely identifiers (IDs/primary keys).

    Combines three signals: identifier-like column names, near-100%
    uniqueness, and sequential-integer patterns (e.g. 1, 2, 3, ...).

    Args:
        df: The dataset to inspect.

    Returns:
        A list of column names judged to be identifier columns.
    """
    identifier_columns: List[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return identifier_columns

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        n_unique = int(non_null.nunique())
        uniqueness_ratio = n_unique / len(non_null)

        name_match = bool(_ID_NAME_PATTERN.search(str(col)))
        is_high_uniqueness = uniqueness_ratio >= 0.98

        is_sequential_int = False
        if pd.api.types.is_integer_dtype(series) and is_high_uniqueness and len(non_null) > 1:
            sorted_vals = non_null.sort_values().reset_index(drop=True)
            diffs = sorted_vals.diff().dropna()
            if len(diffs) > 0:
                is_sequential_int = bool((diffs == 1).mean() > 0.95)

        is_near_unique_string = (
            _column_dtype_kind(series) == "categorical"
            and uniqueness_ratio >= 0.99
            and len(non_null) > 1
        )

        if name_match and uniqueness_ratio >= 0.9:
            identifier_columns.append(col)
        elif is_high_uniqueness and (is_sequential_int or name_match):
            identifier_columns.append(col)
        elif is_near_unique_string:
            identifier_columns.append(col)

    return identifier_columns


#----------------------------------------------------------------------
# Feature analysis
#----------------------------------------------------------------------


def feature_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Profile every column in the dataset.

    Args:
        df: The dataset to analyze.

    Returns:
        A mapping of column name -> profile dict containing dtype_kind,
        pandas_dtype, n_unique, uniqueness_ratio, non_null_count,
        non_null_percentage, and (for numeric columns) basic descriptive
        statistics. All values are JSON-serializable.
    """
    n_rows = len(df)
    result: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])
        n_unique = int(non_null.nunique())
        dtype_kind = _column_dtype_kind(series)

        profile: Dict[str, Any] = {
            "dtype_kind": dtype_kind,
            "pandas_dtype": str(series.dtype),
            "n_unique": n_unique,
            "uniqueness_ratio": round(n_unique / non_null_count, 4) if non_null_count else 0.0,
            "non_null_count": non_null_count,
            "non_null_percentage": round((non_null_count / n_rows) * 100, 2) if n_rows else 0.0,
        }

        if dtype_kind == "numeric" and non_null_count > 0:
            numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
            if not numeric_series.empty:
                profile.update(
                    {
                        "min": float(numeric_series.min()),
                        "max": float(numeric_series.max()),
                        "mean": float(numeric_series.mean()),
                        "std": float(numeric_series.std()) if len(numeric_series) > 1 else 0.0,
                        "is_integer_like": _is_integer_like(series),
                    }
                )
        elif dtype_kind == "categorical" and non_null_count > 0:
            top_values = non_null.astype(str).value_counts().head(5)
            profile["top_values"] = {
                str(k): int(v) for k, v in top_values.items()
            }

        result[str(col)] = profile

    return to_json_safe(result)


#----------------------------------------------------------------------
# Missing value / duplicate / constant-column analysis
#----------------------------------------------------------------------


def missing_value_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute per-column missing value counts and percentages.

    Args:
        df: The dataset to analyze.

    Returns:
        A mapping of column name -> {"missing_count": int,
        "missing_percentage": float}.
    """
    n_rows = len(df)
    result: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_percentage = round((missing_count / n_rows) * 100, 2) if n_rows else 0.0
        result[str(col)] = {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
    return to_json_safe(result)


def duplicate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute duplicate-row statistics for the dataset.

    Args:
        df: The dataset to analyze.

    Returns:
        A dict with "duplicate_row_count" and "duplicate_row_percentage".
    """
    n_rows = len(df)
    duplicate_row_count = int(df.duplicated(keep="first").sum())
    duplicate_row_percentage = (
        round((duplicate_row_count / n_rows) * 100, 2) if n_rows else 0.0
    )
    return to_json_safe(
        {
            "duplicate_row_count": duplicate_row_count,
            "duplicate_row_percentage": duplicate_row_percentage,
        }
    )


def constant_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns that contain only a single distinct (non-null) value.

    Args:
        df: The dataset to analyze.

    Returns:
        List of column names that are constant.
    """
    result: List[str] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if non_null.nunique() == 1:
            result.append(str(col))
    return result


def near_constant_columns(
    df: pd.DataFrame, dominance_threshold: float = 0.95
) -> List[str]:
    """Identify columns where one value dominates almost all rows.

    Args:
        df: The dataset to analyze.
        dominance_threshold: Fraction of non-null rows that the single
            most frequent value must occupy to flag the column.
            Defaults to 0.95.

    Returns:
        List of near-constant column names (excludes fully constant
        columns, which are reported separately by ``constant_columns``).
    """
    result: List[str] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty or non_null.nunique() <= 1:
            continue
        top_freq_ratio = non_null.value_counts(normalize=True).iloc[0]
        if top_freq_ratio >= dominance_threshold:
            result.append(str(col))
    return result


def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a high-level overview of the dataset shape and composition.

    Args:
        df: The dataset to analyze.

    Returns:
        A dict with row_count, column_count, column_names, and a count of
        columns per dtype_kind.
    """
    dtype_counts: Dict[str, int] = {}
    for col in df.columns:
        kind = _column_dtype_kind(df[col])
        dtype_counts[kind] = dtype_counts.get(kind, 0) + 1

    return to_json_safe(
        {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "column_names": [str(c) for c in df.columns],
            "dtype_kind_counts": dtype_counts,
        }
    )


def quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Run a battery of basic data-quality checks.

    Args:
        df: The dataset to analyze.

    Returns:
        A dict summarizing empty columns, fully-duplicated columns, and
        whether the dataset is empty.
    """
    n_rows, n_cols = df.shape
    empty_columns = [str(c) for c in df.columns if df[c].notna().sum() == 0]

    fully_duplicated_columns: List[List[str]] = []
    seen_signatures: Dict[Tuple[Any, ...], str] = {}
    for col in df.columns:
        try:
            signature = tuple(df[col].fillna("__NA__").astype(str).to_numpy())
        except (TypeError, ValueError):
            continue
        if signature in seen_signatures:
            fully_duplicated_columns.append([seen_signatures[signature], str(col)])
        else:
            seen_signatures[signature] = str(col)

    return to_json_safe(
        {
            "is_empty": n_rows == 0 or n_cols == 0,
            "empty_columns": empty_columns,
            "fully_duplicated_column_pairs": fully_duplicated_columns,
        }
    )


#----------------------------------------------------------------------
# ML task inference (core bug fix)
#----------------------------------------------------------------------

_TARGET_NAME_HINTS = [
    "target", "label", "class", "outcome", "result", "y",
    "price", "cost", "value", "amount", "score", "rating",
    "salary", "revenue", "sales", "churn", "fraud", "default",
    "status", "category", "type", "role", "grade", "rank",
    "diagnosis", "approved", "success", "survived", "job_role",
]


def _name_hint_score(col_name: str) -> float:
    """Score a column name against common target-name heuristics.

    Matching is boundary-aware: short hints (e.g. "y") only match on
    exact equality or underscore-delimited tokens, never as a bare
    substring -- otherwise "y" would falsely match inside words like
    "clarity" or "salary". Longer hints (>= 4 chars) are also allowed
    to match as a substring, since false positives there are rare.

    Args:
        col_name: The column name to score.

    Returns:
        1.0 if the name matches a known target-name pattern, else 0.0.
    """
    name = col_name.lower()
    tokens = set(name.split("_"))

    for hint in _TARGET_NAME_HINTS:
        if name == hint or hint in tokens:
            return 1.0
        if name.endswith(f"_{hint}") or name.startswith(f"{hint}_"):
            return 1.0
        if len(hint) >= 4 and hint in name:
            return 1.0
    return 0.0


def _infer_task_for_series(series: pd.Series, n_rows: int) -> Tuple[str, float]:
    """Infer the ML task type implied by treating a series as the target.

    Args:
        series: Candidate target column.
        n_rows: Total number of rows in the dataset (for cardinality
            ratio heuristics).

    Returns:
        A (task_type, confidence) tuple where task_type is one of
        "classification", "regression", or "unknown", and confidence is
        a float in [0, 1].
    """
    non_null = series.dropna()
    if non_null.empty or n_rows == 0:
        return "unknown", 0.0

    n_unique = int(non_null.nunique())
    uniqueness_ratio = n_unique / len(non_null)
    kind = _column_dtype_kind(series)

    if kind == "boolean":
        return "classification", 0.95

    if kind == "categorical":
        # Bounded label space relative to dataset size -> classification.
        cardinality_cap = max(50, int(0.2 * n_rows))
        if n_unique <= cardinality_cap:
            confidence = 0.95 if n_unique <= 20 else 0.8
            return "classification", confidence
        # Free-text-like column with near-unique values is not a usable
        # target (e.g. names, descriptions, IDs that slipped through).
        return "unknown", 0.3

    if kind == "numeric":
        if n_unique <= 2:
            return "classification", 0.9
        if _is_integer_like(series) and n_unique <= 20 and uniqueness_ratio <= 0.3:
            return "classification", 0.75
        if n_unique > 20:
            return "regression", 0.9
        return "regression", 0.6

    if kind == "datetime":
        return "unknown", 0.2

    return "unknown", 0.1


def infer_ml_tasks(
    df: pd.DataFrame, identifier_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Infer the most likely ML target column and its task type.

    Candidate target columns are ranked by a weighted combination of:
    task-type confidence (dtype/cardinality heuristics), column-name
    heuristics, "last column" position heuristics, and data completeness.
    Identifier columns are always excluded from consideration.

    Args:
        df: The dataset to analyze.
        identifier_columns: Optional pre-computed identifier columns to
            exclude. If ``None``, they are computed internally via
            ``detect_identifier_columns``.

    Returns:
        A dict with "likely_target", "task_type", "confidence"
        (categorical label), "confidence_score" (float in [0, 1]), and
        "candidates" (top-5 ranked candidates with their scores).
    """
    if identifier_columns is None:
        identifier_columns = detect_identifier_columns(df)

    n_rows = len(df)
    if n_rows == 0 or df.shape[1] == 0:
        return to_json_safe(
            {
                "likely_target": None,
                "task_type": "unknown",
                "confidence": "low",
                "confidence_score": 0.0,
                "candidates": [],
            }
        )

    candidates: List[Dict[str, Any]] = []

    for col in df.columns:
        if col in identifier_columns:
            continue

        series = df[col]
        non_null_ratio = float(series.notna().mean())
        if non_null_ratio == 0:
            continue

        task_type, base_confidence = _infer_task_for_series(series, n_rows)

        name_score = _name_hint_score(str(col))
        position_score = 1.0 if col == df.columns[-1] else 0.0

        if task_type == "unknown":
            priority = 0.05 * non_null_ratio
        else:
            priority = (
                0.5 * base_confidence
                + 0.35 * name_score
                + 0.1 * position_score
                + 0.05 * non_null_ratio
            )

        candidates.append(
            {
                "column": str(col),
                "task_type": task_type,
                "confidence_score": round(float(base_confidence), 3),
                "priority": round(float(priority), 3),
            }
        )

    if not candidates:
        return to_json_safe(
            {
                "likely_target": None,
                "task_type": "unknown",
                "confidence": "low",
                "confidence_score": 0.0,
                "candidates": [],
            }
        )

    candidates.sort(key=lambda c: c["priority"], reverse=True)
    best = candidates[0]

    confidence_score = best["confidence_score"]
    if best["task_type"] == "unknown":
        confidence_label = "low"
    elif confidence_score >= 0.8:
        confidence_label = "high"
    elif confidence_score >= 0.5:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    return to_json_safe(
        {
            "likely_target": best["column"],
            "task_type": best["task_type"],
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "candidates": candidates[:5],
        }
    )


#----------------------------------------------------------------------
# Project recommendations
#----------------------------------------------------------------------


def project_recommendations(
    task_type: str,
    dataset_size: int,
    feature_types: Dict[str, int],
    target_type: Optional[str] = None,
) -> List[str]:
    """Generate project ideas tailored to the dataset's task and composition.

    Args:
        task_type: "classification", "regression", or "unknown".
        dataset_size: Number of rows in the dataset.
        feature_types: Mapping of dtype_kind -> count (e.g. output of
            ``dataset_overview()["dtype_kind_counts"]``).
        target_type: Optional dtype_kind of the chosen target column.

    Returns:
        An ordered list of recommended project ideas, de-duplicated.
    """
    recommendations: List[str] = []

    if task_type == "regression":
        recommendations.extend(
            [
                "Regression Modeling",
                "Feature Importance Analysis",
                "Prediction Dashboard",
                "Error Analysis",
            ]
        )
    elif task_type == "classification":
        recommendations.extend(
            [
                "Classification Modeling",
                "Explainable AI (SHAP / LIME)",
                "Feature Importance Analysis",
                "Model Comparison Benchmark",
            ]
        )
    else:
        recommendations.extend(
            [
                "Exploratory Data Analysis",
                "Clustering / Unsupervised Pattern Discovery",
                "Anomaly Detection",
            ]
        )

    numeric_count = feature_types.get("numeric", 0)
    categorical_count = feature_types.get("categorical", 0)
    datetime_count = feature_types.get("datetime", 0)

    if datetime_count > 0:
        recommendations.append("Time Series Trend Analysis")
    if categorical_count > numeric_count and categorical_count >= 3:
        recommendations.append("Categorical Encoding Strategy Comparison")
    if dataset_size >= 50_000:
        recommendations.append("Scalable Pipeline / Batch Inference Design")
    elif 0 < dataset_size < 500:
        recommendations.append("Data Augmentation / Small-Sample Strategies")

    recommendations.append("Streamlit Deployment")

    seen: set[str] = set()
    deduped: List[str] = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            deduped.append(rec)
    return deduped


#----------------------------------------------------------------------
# Preprocessing recommendations
#----------------------------------------------------------------------


def generate_preprocessing_recommendations(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate preprocessing recommendations from a completed analysis.

    All sections are read defensively with ``.get()``, so this function
    degrades gracefully if ``analysis_results`` only contains a subset of
    the full schema produced by ``run_full_analysis``.

    Expected (optional) keys in ``analysis_results``:
        - "identifier_columns": List[str]
        - "feature_analysis": Dict[str, Dict[str, Any]]
        - "missing_value_analysis": Dict[str, Dict[str, Any]]
        - "duplicate_analysis": Dict[str, Any]
        - "constant_columns": List[str]
        - "near_constant_columns": List[str]
        - "ml_task_inference": Dict[str, Any]

    Args:
        analysis_results: Combined output of the other analysis functions.

    Returns:
        A dict with a single "recommendations" key holding an ordered
        list of human-readable preprocessing suggestions.
    """
    recommendations: List[str] = []

    identifier_columns = analysis_results.get("identifier_columns", []) or []
    for col in identifier_columns:
        recommendations.append(f"Remove identifier column: {col}")

    feature_analysis_results = analysis_results.get("feature_analysis", {}) or {}
    for col, info in feature_analysis_results.items():
        if col in identifier_columns:
            continue
        dtype_kind = info.get("dtype_kind")
        n_unique = info.get("n_unique", 0)
        if dtype_kind == "categorical" and isinstance(n_unique, (int, float)) and n_unique > 50:
            recommendations.append(f"Handle high-cardinality feature: {col}")

    numeric_feature_cols = [
        col
        for col, info in feature_analysis_results.items()
        if info.get("dtype_kind") == "numeric" and col not in identifier_columns
    ]
    if numeric_feature_cols:
        recommendations.append("Consider feature scaling for numerical variables")

    missing_value_results = analysis_results.get("missing_value_analysis", {}) or {}
    for col, info in missing_value_results.items():
        pct = info.get("missing_percentage", 0)
        if isinstance(pct, (int, float)) and pct > 0:
            if pct >= 30:
                recommendations.append(
                    f"Consider dropping or carefully imputing high-missingness "
                    f"column: {col} ({round(float(pct), 1)}% missing)"
                )
            else:
                recommendations.append(f"Impute missing values in column: {col}")

    duplicate_results = analysis_results.get("duplicate_analysis", {}) or {}
    duplicate_count = duplicate_results.get("duplicate_row_count", 0) or 0
    if isinstance(duplicate_count, (int, float)) and duplicate_count > 0:
        recommendations.append(f"Remove {int(duplicate_count)} duplicate row(s)")

    constant_cols = analysis_results.get("constant_columns", []) or []
    for col in constant_cols:
        recommendations.append(f"Drop constant column: {col}")

    near_constant_cols = analysis_results.get("near_constant_columns", []) or []
    for col in near_constant_cols:
        recommendations.append(f"Review near-constant column (low variance): {col}")

    ml_task_inference = analysis_results.get("ml_task_inference", {}) or {}
    task_type = ml_task_inference.get("task_type")
    if task_type == "classification":
        recommendations.append(
            "Check class balance and consider stratified sampling or class weighting"
        )
    elif task_type == "regression":
        recommendations.append(
            "Inspect target distribution for skewness; consider a log-transform if heavily skewed"
        )

    return {"recommendations": recommendations}


#----------------------------------------------------------------------
# Dataset scoring
#----------------------------------------------------------------------


def dataset_score(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute an overall dataset quality score (0-10) with a breakdown.

    Score components (sum to a 10.0 ceiling before penalties):
        - size_score (0-2.0): rewards larger datasets (log-scaled).
        - missingness_score (0-2.5): penalizes high average missingness.
        - duplicates_score (0-1.5): penalizes duplicate rows.
        - cleanliness_score (0-1.5): penalizes constant/near-constant cols.
        - target_detection_score (0-2.5): rewards confident target/task
          detection.
    Penalties:
        - identifier_penalty (0-1.0): penalizes presence of identifier
          columns, since they typically must be dropped before modeling.

    Args:
        analysis_results: Combined output of the other analysis functions,
            expected to (optionally) include "dataset_overview",
            "missing_value_analysis", "duplicate_analysis",
            "constant_columns", "near_constant_columns",
            "ml_task_inference", and "identifier_columns".

    Returns:
        A dict with "overall_score" (float), "recommendation" (str), and
        "breakdown" (Dict[str, float]).
    """
    breakdown: Dict[str, float] = {}

    overview = analysis_results.get("dataset_overview", {}) or {}
    row_count = overview.get("row_count", 0) or 0
    column_count = max(1, overview.get("column_count", 1) or 1)

    if row_count <= 0:
        return to_json_safe(
            {
                "overall_score": 0.0,
                "recommendation": "Not recommended",
                "breakdown": {
                    "size_score": 0.0,
                    "missingness_score": 0.0,
                    "duplicates_score": 0.0,
                    "cleanliness_score": 0.0,
                    "target_detection_score": 0.0,
                    "identifier_penalty": 0.0,
                },
            }
        )

    size_score = min(2.0, 2.0 * (np.log10(row_count + 1) / 5.0))
    breakdown["size_score"] = round(float(size_score), 3)

    missing_value_results = analysis_results.get("missing_value_analysis", {}) or {}
    if missing_value_results:
        pct_values = [
            float(info.get("missing_percentage", 0.0))
            for info in missing_value_results.values()
            if isinstance(info.get("missing_percentage", 0.0), (int, float))
        ]
        avg_missing_pct = sum(pct_values) / len(pct_values) if pct_values else 0.0
    else:
        avg_missing_pct = 0.0
    missingness_score = max(0.0, 2.5 * (1 - avg_missing_pct / 100.0))
    breakdown["missingness_score"] = round(float(missingness_score), 3)

    duplicate_results = analysis_results.get("duplicate_analysis", {}) or {}
    duplicate_count = duplicate_results.get("duplicate_row_count", 0) or 0
    duplicate_ratio = (duplicate_count / row_count) if row_count else 0.0
    duplicates_score = max(0.0, 1.5 * (1 - min(duplicate_ratio, 1.0)))
    breakdown["duplicates_score"] = round(float(duplicates_score), 3)

    constant_cols = analysis_results.get("constant_columns", []) or []
    near_constant_cols = analysis_results.get("near_constant_columns", []) or []
    problem_ratio = (len(constant_cols) + 0.5 * len(near_constant_cols)) / column_count
    cleanliness_score = max(0.0, 1.5 * (1 - min(problem_ratio, 1.0)))
    breakdown["cleanliness_score"] = round(float(cleanliness_score), 3)

    ml_task_inference = analysis_results.get("ml_task_inference", {}) or {}
    confidence_score = float(ml_task_inference.get("confidence_score", 0.0) or 0.0)
    task_type = ml_task_inference.get("task_type", "unknown")
    target_detection_score = 2.5 * confidence_score if task_type != "unknown" else 0.0
    breakdown["target_detection_score"] = round(float(target_detection_score), 3)

    identifier_columns = analysis_results.get("identifier_columns", []) or []
    identifier_penalty = min(1.0, 0.3 * len(identifier_columns))
    breakdown["identifier_penalty"] = round(float(identifier_penalty), 3)

    raw_total = (
        size_score
        + missingness_score
        + duplicates_score
        + cleanliness_score
        + target_detection_score
        - identifier_penalty
    )
    overall_score = max(0.0, min(10.0, raw_total))

    if overall_score >= 8.0:
        recommendation = "Recommended"
    elif overall_score >= 6.0:
        recommendation = "Usable with caution"
    elif overall_score >= 4.0:
        recommendation = "Needs significant cleanup"
    else:
        recommendation = "Not recommended"

    return to_json_safe(
        {
            "overall_score": round(float(overall_score), 2),
            "recommendation": recommendation,
            "breakdown": breakdown,
        }
    )


#----------------------------------------------------------------------
# Report generation (deterministic, no LLM calls)
#----------------------------------------------------------------------


def generate_report(analysis_results: Dict[str, Any]) -> str:
    """Generate a clean, human-readable text report from analysis results.

    Purely deterministic string formatting -- no LLM calls are made.

    Args:
        analysis_results: Combined output of ``run_full_analysis`` (or any
            dict with a compatible subset of keys: "dataset_score",
            "ml_task_inference", "missing_value_analysis",
            "duplicate_analysis", "identifier_columns",
            "constant_columns", "near_constant_columns",
            "project_recommendations", "preprocessing_recommendations").

    Returns:
        A formatted multi-line report string.
    """
    lines: List[str] = []

    score_info = analysis_results.get("dataset_score", {}) or {}
    overall_score = score_info.get("overall_score")
    recommendation = score_info.get("recommendation", "Unknown")

    ml_task_inference = analysis_results.get("ml_task_inference", {}) or {}
    task_type = ml_task_inference.get("task_type", "unknown")
    likely_target = ml_task_inference.get("likely_target")

    lines.append("=" * 50)
    lines.append("DATASET REPORT")
    lines.append("=" * 50)
    lines.append("")

    if isinstance(overall_score, (int, float)):
        lines.append(f"Overall Score: {round(float(overall_score), 2)}/10")
    else:
        lines.append("Overall Score: N/A")
    lines.append("")

    lines.append("Recommendation:")
    lines.append(recommendation)
    lines.append("")

    lines.append("Likely Task:")
    lines.append(task_type.capitalize() if isinstance(task_type, str) else "Unknown")
    lines.append("")

    lines.append("Likely Target:")
    lines.append(likely_target if likely_target else "Not detected")
    lines.append("")

    strengths: List[str] = []
    warnings: List[str] = []

    missing_value_results = analysis_results.get("missing_value_analysis", {}) or {}
    total_missing = sum(
        float(info.get("missing_percentage", 0.0))
        for info in missing_value_results.values()
        if isinstance(info.get("missing_percentage", 0.0), (int, float))
    )
    if total_missing == 0:
        strengths.append("No missing values")
    else:
        warnings.append("Missing values detected in one or more columns")

    duplicate_results = analysis_results.get("duplicate_analysis", {}) or {}
    duplicate_count = duplicate_results.get("duplicate_row_count", 0) or 0
    if duplicate_count == 0:
        strengths.append("No duplicate rows")
    else:
        warnings.append(f"{int(duplicate_count)} duplicate row(s) detected")

    identifier_columns = analysis_results.get("identifier_columns", []) or []
    if identifier_columns:
        warnings.append("Potential identifier column detected")
    else:
        strengths.append("No identifier columns detected")

    constant_cols = analysis_results.get("constant_columns", []) or []
    if constant_cols:
        warnings.append(f"Constant column(s) detected: {', '.join(constant_cols)}")

    near_constant_cols = analysis_results.get("near_constant_columns", []) or []
    if near_constant_cols:
        warnings.append(f"Near-constant column(s) detected: {', '.join(near_constant_cols)}")

    if ml_task_inference.get("confidence") == "low":
        warnings.append("Low confidence in target/task detection")

    lines.append("Strengths:")
    if strengths:
        lines.extend(f"\u2713 {item}" for item in strengths)
    else:
        lines.append("(none identified)")
    lines.append("")

    lines.append("Warnings:")
    if warnings:
        lines.extend(f"\u26a0 {item}" for item in warnings)
    else:
        lines.append("(none identified)")
    lines.append("")

    suggested_projects = analysis_results.get("project_recommendations", []) or []
    lines.append("Suggested Projects:")
    if suggested_projects:
        lines.extend(f"\u2022 {item}" for item in suggested_projects)
    else:
        lines.append("(none generated)")
    lines.append("")

    preprocessing_results = analysis_results.get("preprocessing_recommendations", {}) or {}
    preprocessing_items = preprocessing_results.get("recommendations", []) or []
    lines.append("Preprocessing Suggestions:")
    if preprocessing_items:
        lines.extend(f"\u2022 {item}" for item in preprocessing_items)
    else:
        lines.append("(none generated)")

    return "\n".join(lines)


#----------------------------------------------------------------------
# Top-level orchestration
#----------------------------------------------------------------------


def load_dataset(path: str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a tabular dataset from disk into a pandas DataFrame.

    Supports CSV, TSV, JSON, and Parquet based on the file extension.

    Args:
        path: Filesystem path to the dataset.
        **read_csv_kwargs: Extra keyword arguments forwarded to the
            underlying pandas reader (only used for CSV/TSV).

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If the path does not exist.
    """
    lowered = path.lower()
    if lowered.endswith(".csv"):
        return pd.read_csv(path, **read_csv_kwargs)
    if lowered.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", **read_csv_kwargs)
    if lowered.endswith(".json"):
        return pd.read_json(path)
    if lowered.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file extension for '{path}'. "
        "Supported formats: .csv, .tsv, .json, .parquet"
    )


def run_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the complete Dataset analysis pipeline on a DataFrame.

    Args:
        df: The dataset to analyze.

    Returns:
        A single JSON-serializable dict aggregating every analysis stage:
        dataset_overview, feature_analysis, missing_value_analysis,
        duplicate_analysis, constant_columns, near_constant_columns,
        identifier_columns, ml_task_inference, project_recommendations,
        preprocessing_recommendations, dataset_score, and report.
    """
    overview = dataset_overview(df)
    features = feature_analysis(df)
    missing = missing_value_analysis(df)
    duplicates = duplicate_analysis(df)
    constants = constant_columns(df)
    near_constants = near_constant_columns(df)
    identifiers = detect_identifier_columns(df)
    ml_task = infer_ml_tasks(df, identifier_columns=identifiers)
    quality = quality_checks(df)

    projects = project_recommendations(
        task_type=ml_task["task_type"],
        dataset_size=overview["row_count"],
        feature_types=overview["dtype_kind_counts"],
        target_type=features.get(ml_task["likely_target"], {}).get("dtype_kind")
        if ml_task.get("likely_target")
        else None,
    )

    partial_results: Dict[str, Any] = {
        "dataset_overview": overview,
        "feature_analysis": features,
        "missing_value_analysis": missing,
        "duplicate_analysis": duplicates,
        "constant_columns": constants,
        "near_constant_columns": near_constants,
        "identifier_columns": identifiers,
        "ml_task_inference": ml_task,
        "quality_checks": quality,
        "project_recommendations": projects,
    }

    preprocessing = generate_preprocessing_recommendations(partial_results)
    partial_results["preprocessing_recommendations"] = preprocessing

    score = dataset_score(partial_results)
    partial_results["dataset_score"] = score

    report = generate_report(partial_results)
    partial_results["report"] = report

    return to_json_safe(partial_results)