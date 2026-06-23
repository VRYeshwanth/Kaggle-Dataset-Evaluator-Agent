from pathlib import Path
import json
from io import BytesIO

import pandas as pd
import streamlit as st

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet

from mcp_server.tools import analyze_dataset

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Kaggle Dataset Evaluator",
    page_icon="📊",
    layout="wide",
)

# ======================================================
# SESSION STATE
# ======================================================

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ======================================================
# HELPERS
# ======================================================

def save_uploaded_file(uploaded_file):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


@st.cache_data(show_spinner=False)
def run_analysis_cached(dataset_path: str):
    return analyze_dataset(dataset_path)


def create_feature_dataframe(feature_analysis):
    rows = []

    for feature, info in feature_analysis.items():
        rows.append(
            {
                "Feature": feature,
                "Type": info.get("dtype_kind"),
                "Unique Values": info.get("n_unique"),
                "Uniqueness Ratio": info.get(
                    "uniqueness_ratio"
                ),
                "Non Null %": info.get(
                    "non_null_percentage"
                ),
            }
        )

    return pd.DataFrame(rows)


def create_missing_dataframe(missing):
    rows = []

    for column, info in missing.items():
        rows.append(
            {
                "Column": column,
                "Missing Count": info.get(
                    "missing_count"
                ),
                "Missing %": info.get(
                    "missing_percentage"
                ),
            }
        )

    return pd.DataFrame(rows)

def get_score_status(score):

    if score >= 8:
        return "🟢 Excellent"

    elif score >= 6:
        return "🟡 Good"

    elif score >= 4:
        return "🟠 Needs Cleanup"

    return "🔴 Poor"


def generate_pdf_report(report_text):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "Dataset Analysis Report",
            styles["Title"],
        )
    )

    content.append(
        Spacer(1, 12)
    )

    for line in report_text.split("\n"):

        if line.strip():

            content.append(
                Paragraph(
                    line,
                    styles["BodyText"],
                )
            )

    doc.build(content)

    buffer.seek(0)

    return buffer


# ======================================================
# HEADER
# ======================================================

st.title("📊 Kaggle Dataset Evaluator Agent")

st.caption(
    "Evaluate datasets before investing time into machine learning projects."
)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
)

# ======================================================
# FILE HANDLING
# ======================================================

if uploaded_file:

    if (
        st.session_state.current_file
        != uploaded_file.name
    ):
        st.session_state.analysis_results = None
        st.session_state.current_file = (
            uploaded_file.name
        )

    dataset_path = save_uploaded_file(
        uploaded_file
    )

    try:
        preview_df = pd.read_csv(
            dataset_path,
            nrows=20,
        )
    except Exception:
        preview_df = None

    st.sidebar.success(
        f"Loaded: {uploaded_file.name}"
    )

    if st.sidebar.button(
        "Analyze Dataset",
        use_container_width=True,
    ):
        with st.spinner(
            "Running analysis..."
        ):
            st.session_state.analysis_results = (
                run_analysis_cached(
                    str(dataset_path)
                )
            )

# ======================================================
# RESULTS
# ======================================================

results = st.session_state.analysis_results

if results:

    if "error" in results:
        st.error(results["error"])
        st.stop()

    # ==================================================
    # EXTRACT DATA
    # ==================================================

    overview = results[
        "dataset_overview"
    ]

    score = results[
        "dataset_score"
    ]

    ml_task = results[
        "ml_task_inference"
    ]

    # ==================================================
    # SUMMARY CARDS
    # ==================================================

    st.subheader(
    "Dataset Preview"
    )

    if preview_df is not None:

        st.dataframe(
            preview_df,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "Dataset Summary"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    overall_score = score.get(
        "overall_score",
        0,
    )

    c1.metric(
        "Score",
        overall_score,
    )

    c2.metric(
        "Status",
        get_score_status(
            overall_score
        ),
    )

    c3.metric(
        "Rows",
        overview.get(
            "row_count",
            0,
        ),
    )

    c4.metric(
        "Columns",
        overview.get(
            "column_count",
            0,
        ),
    )

    c5.metric(
        "Task",
        ml_task.get(
            "task_type",
            "Unknown",
        ),
    )

    c6.metric(
        "Target",
        ml_task.get(
            "likely_target",
            "Unknown",
        ),
    )

    st.progress(
        score.get(
            "overall_score",
            0,
        )
        / 10
    )

    # ==================================================
    # TABS
    # ==================================================

    (
        tab1,
        tab2,
        tab3,
        tab4,
        tab5,
    ) = st.tabs(
        [
            "Overview",
            "Features",
            "Data Quality",
            "ML Insights",
            "Report",
        ]
    )

    # ==================================================
    # OVERVIEW
    # ==================================================

    with tab1:

        st.subheader(
            "Dataset Overview"
        )

        overview_df = pd.DataFrame(
            {
                "Property": [
                    "Rows",
                    "Columns",
                ],
                "Value": [
                    overview["row_count"],
                    overview["column_count"],
                ],
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
        )

        st.subheader(
            "Feature Type Distribution"
        )

        dtype_counts = overview.get(
            "dtype_kind_counts",
            {}
        )

        if dtype_counts:

            chart_df = pd.DataFrame(
                {
                    "Type": dtype_counts.keys(),
                    "Count": dtype_counts.values(),
                }
            )

            st.bar_chart(
                chart_df.set_index(
                    "Type"
                )
            )

        st.subheader(
            "Column Names"
        )

        st.write(
            overview.get(
                "column_names",
                []
            )
        )

    # ==================================================
    # FEATURES
    # ==================================================

    with tab2:

        st.subheader(
            "Feature Summary"
        )

        feature_df = (
            create_feature_dataframe(
                results[
                    "feature_analysis"
                ]
            )
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
        )

        st.subheader(
            "Numeric Features"
        )

        numeric_df = feature_df[
            feature_df["Type"]
            == "numeric"
        ]

        st.dataframe(
            numeric_df,
            use_container_width=True,
        )

        st.subheader(
            "Categorical Features"
        )

        categorical_df = feature_df[
            feature_df["Type"]
            == "categorical"
        ]

        st.dataframe(
            categorical_df,
            use_container_width=True,
        )

    # ==================================================
    # DATA QUALITY
    # ==================================================

    with tab3:

        st.subheader(
            "Missing Value Analysis"
        )

        missing_df = (
            create_missing_dataframe(
                results[
                    "missing_value_analysis"
                ]
            )
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
        )

        st.subheader(
            "Quality Metrics"
        )

        strengths = []
        weaknesses = []

        if len(results["constant_columns"]) == 0:
            strengths.append(
                "No constant columns"
            )

        if len(results["identifier_columns"]) == 0:
            strengths.append(
                "No identifier columns"
            )

        duplicate_count = (
            results["duplicate_analysis"]
            .get(
                "duplicate_row_count",
                0,
            )
        )

        if duplicate_count == 0:
            strengths.append(
                "No duplicate rows"
            )

        missing_df_temp = (
            create_missing_dataframe(
                results[
                    "missing_value_analysis"
                ]
            )
        )

        if (
            missing_df_temp[
                "Missing Count"
            ].sum()
            == 0
        ):
            strengths.append(
                "No missing values"
            )

        if len(results["identifier_columns"]) > 0:
            weaknesses.append(
                "Identifier columns detected"
            )

        if len(results["near_constant_columns"]) > 0:
            weaknesses.append(
                "Near-constant columns detected"
            )

        st.markdown(
            "### Dataset Health Summary"
        )

        col_a, col_b = st.columns(2)

        with col_a:

            st.success(
                "\n".join(
                    [
                        f"✓ {s}"
                        for s in strengths
                    ]
                )
                if strengths
                else "No notable strengths"
            )

        with col_b:

            st.warning(
                "\n".join(
                    [
                        f"⚠ {w}"
                        for w in weaknesses
                    ]
                )
                if weaknesses
                else "No notable issues"
            )

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Constant Columns",
            len(
                results[
                    "constant_columns"
                ]
            ),
        )

        q2.metric(
            "Near Constant",
            len(
                results[
                    "near_constant_columns"
                ]
            ),
        )

        q3.metric(
            "Identifier Columns",
            len(
                results[
                    "identifier_columns"
                ]
            ),
        )

        st.subheader(
            "Problem Columns"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                "#### Constant"
            )
            st.write(
                results[
                    "constant_columns"
                ]
            )

        with col2:
            st.markdown(
                "#### Near Constant"
            )
            st.write(
                results[
                    "near_constant_columns"
                ]
            )

        with col3:
            st.markdown(
                "#### Identifiers"
            )
            st.write(
                results[
                    "identifier_columns"
                ]
            )

        st.subheader(
            "Duplicate Analysis"
        )

        duplicate_df = pd.DataFrame(
            [
                results[
                    "duplicate_analysis"
                ]
            ]
        )

        st.dataframe(
            duplicate_df,
            use_container_width=True,
        )

    # ==================================================
    # ML INSIGHTS
    # ==================================================

    with tab4:

        st.subheader(
            "ML Task Inference"
        )

        st.json(
            results[
                "ml_task_inference"
            ]
        )

        st.subheader(
            "Dataset Score Breakdown"
        )

        breakdown = score.get(
            "breakdown",
            {}
        )

        if breakdown:

            breakdown_df = pd.DataFrame(
                {
                    "Component":
                    breakdown.keys(),
                    "Score":
                    breakdown.values(),
                }
            )

            st.dataframe(
                breakdown_df,
                use_container_width=True,
            )

            st.bar_chart(
                breakdown_df.set_index(
                    "Component"
                )
            )

        st.subheader(
            "Project Recommendations"
        )

        recommendations = results.get(
            "project_recommendations",
            []
        )

        if isinstance(
            recommendations,
            list,
        ):
            for rec in recommendations:
                st.success(rec)
        else:
            st.json(
                recommendations
            )

        st.subheader(
            "Preprocessing Recommendations"
        )

        preprocessing = results.get(
            "preprocessing_recommendations",
            []
        )

        if isinstance(
            preprocessing,
            list,
        ):
            for item in preprocessing:
                st.info(item)
        else:
            st.json(
                preprocessing
            )

    # ==================================================
    # REPORT
    # ==================================================

    with tab5:

        st.subheader(
            "Generated Report"
        )

        report = results.get(
            "report",
            ""
        )

        st.text_area(
            "Report",
            report,
            height=500,
        )

        download_col1, download_col2 = (
            st.columns(2)
        )

        with download_col1:

            pdf_buffer = generate_pdf_report(report)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="dataset_report.pdf",
                mime="application/pdf",
            )

        with download_col2:

            st.download_button(
                label="📦 Download Analysis JSON",
                data=json.dumps(
                    results,
                    indent=2,
                ),
                file_name="analysis.json",
                mime="application/json",
            )

else:

    st.info(
        "Upload a dataset and click 'Analyze Dataset' to begin."
    )