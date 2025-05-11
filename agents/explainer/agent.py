import os
from google.adk.agents import Agent
from google.adk.models import LiteLlm
from dotenv import load_dotenv

load_dotenv()

def create_agent():
    return Agent(
        name="explainer",
        description="Generates explanations using Gemini",
        model=LiteLlm(model="gemini-1.5-pro-latest", api_key=os.getenv("GOOGLE_API_KEY")),
        instruction="Provide clear explanations with examples. Break down complex concepts."
    )