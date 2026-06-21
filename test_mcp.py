# test_mcp.py

import asyncio
import sys

from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)


async def main():

    toolset = MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_server.server"],
            )
        )
    )

    tools = await toolset.get_tools()

    print("\nTOOLS FOUND:\n")

    for tool in tools:
        print(tool.name)


asyncio.run(main())