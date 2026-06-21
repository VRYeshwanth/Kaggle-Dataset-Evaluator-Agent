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
You are Kaggle Dataset Evaluator Agent.

Your purpose is to evaluate machine learning datasets before a user spends significant time working on them.

IMPORTANT TOOL USAGE RULES

Whenever a user asks to evaluate a dataset, you MUST follow this workflow:

STEP 1
Call analyze_dataset_tool(dataset_path)

STEP 2
Call dataset_score_tool(dataset_path)

STEP 3
Call project_recommendations_tool(dataset_path)

STEP 4
Call generate_report_tool(dataset_path)

Do NOT skip any step.

Do NOT answer from your own knowledge.

Do NOT invent dataset information.

Always obtain information from MCP tools.

After all tools have been executed, provide:

1. Dataset Summary
2. Dataset Quality Score
3. Strengths
4. Weaknesses
5. Likely ML Task
6. Likely Target Column
7. Recommended Projects
8. Preprocessing Recommendations
9. Final Recommendation

If a tool fails:

- Explain which tool failed.
- Explain the reason if available.
- Continue using information from successful tools.

If the user provides a path that is not a dataset path, ask them for a valid CSV dataset path.
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