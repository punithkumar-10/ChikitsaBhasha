import os
import aiohttp
from google.adk.agents import Agent
from google.adk.tools import Tool
from dotenv import load_dotenv

load_dotenv()

class MarkdownTool(Tool):
    name = "markdown_formatter"
    description = "Formats content as Markdown"

    async def __call__(self, content: str):
        # Simple markdown formatting (can be replaced with API call if needed)
        return {
            "formatted": f"# Study Notes\n\n{content.replace('\n', '\n\n')}",
            "download_link": "notes.md"  # Simulated download link
        }

def create_agent():
    return Agent(
        name="note_formatter",
        description="Formats learning content as Markdown notes",
        model="gemini-1.5-flash-latest",
        instruction=(
            "Convert content into well-structured Markdown:\n"
            "1. Use headings and bullet points\n"
            "2. Highlight key concepts\n"
            "3. Generate downloadable .md file"
        ),
        tools=[MarkdownTool()]
    )