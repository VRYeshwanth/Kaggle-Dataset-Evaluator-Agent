"""
agent.py
Command: adk web

Kaggle Dataset Evaluator Agent
"""

import sys

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)

SYSTEM_PROMPT = """
DATASET ANALYSIS RULES

1. If analysis results are already provided in the conversation context, use those results to answer questions.

2. Do NOT call MCP tools if the answer can be derived from existing analysis results.

3. Only execute MCP tools when:
   - Analysis results are not available.
   - Analysis results are incomplete.
   - The user explicitly requests a fresh analysis.
   - The user asks a question requiring information not present in the analysis results.

4. When a fresh dataset evaluation is required, execute the following workflow:

   STEP 1
   Call analyze_dataset_tool(dataset_path)

   STEP 2
   Call dataset_score_tool(dataset_path)

   STEP 3
   Call project_recommendations_tool(dataset_path)

   STEP 4
   Call generate_report_tool(dataset_path)

5. Do not skip any step when performing a fresh evaluation.

6. Prefer existing analysis results over re-running tools.
"""

mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["-m", "mcp_server.server"],
        )
    )
)

root_agent = Agent(
    name="dataset_evaluator_agent",
    model="gemini-2.5-flash",
    description="Evaluates machine learning datasets using MCP tools.",
    instruction=SYSTEM_PROMPT,
    tools=[mcp_toolset],
)