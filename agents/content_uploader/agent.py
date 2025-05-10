import os
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from dotenv import load_dotenv

load_dotenv()

async def create_agent():
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="uv",
            args=["--directory", "./mcp-needle", "run", "needle-mcp"],
            env={"NEEDLE_API_KEY": os.getenv("NEEDLE_API_KEY")}
        )
    )

    return Agent(
        name="content_uploader",
        description="Processes PDFs/PPTs using Needle MCP",
        model="gemini-1.5-flash-latest",
        instruction="Extract text, generate embeddings, and store in vector DB",
        tools=tools
    ), exit_stack