import os
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from dotenv import load_dotenv

load_dotenv()

async def create_agent():
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="npx",
            args=["-y", "@getrember/mcp", f"--api-key={os.getenv('REMBER_API_KEY')}"]
        )
    )

    return Agent(
        name="flashcard_agent",
        description="Creates flashcards with Rember MCP",
        model="gemini-1.5-flash-latest",
        instruction="Generate spaced-repetition flashcards from key concepts",
        tools=tools
    ), exit_stack