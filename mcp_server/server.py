"""
server.py
Command: python -m mcp_server.server

Kaggle Dataset Evaluator Agent MCP Server.
"""

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    analyze_dataset,
    get_dataset_score,
    get_project_recommendations,
    generate_dataset_report,
)

mcp = FastMCP(
    "Kaggle Dataset Evaluator Agent"
)


@mcp.tool()
def analyze_dataset_tool(dataset_path: str) -> dict:
    print("Received:", repr(dataset_path))
    return analyze_dataset(dataset_path)


@mcp.tool()
def dataset_score_tool(
    dataset_path: str,
) -> dict:
    """
    Return dataset score.
    """

    return get_dataset_score(
        dataset_path
    )


@mcp.tool()
def project_recommendations_tool(
    dataset_path: str,
) -> dict:
    """
    Return project ideas.
    """

    return get_project_recommendations(
        dataset_path
    )


@mcp.tool()
def generate_report_tool(
    dataset_path: str,
) -> dict:
    """
    Generate report.
    """

    return generate_dataset_report(
        dataset_path
    )


if __name__ == "__main__":
    mcp.run()