import os
from google.adk.agents import Agent
from google.adk.models import LiteLlm
from dotenv import load_dotenv

load_dotenv()

def create_agent():
    return Agent(
        name="quiz_generator",
        description="Generates quizzes from learning content",
        model=LiteLlm(model="gemini-1.5-pro-latest", api_key=os.getenv("GOOGLE_API_KEY")),
        instruction=(
            "Create quizzes based on learning materials with:\n"
            "1. 3-5 multiple choice questions\n"
            "2. Clear correct answers\n"
            "3. Difficulty based on user level\n"
            "Format: \n"
            "Q1. [Question]\n"
            "A) [Option1]\n"
            "B) [Option2]\n"
            "...\n"
            "Answer: [CorrectOption]"
        )
    )