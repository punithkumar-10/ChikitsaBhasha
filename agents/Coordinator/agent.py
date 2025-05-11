import os
from google.adk.agents import Agent
from google.adk.models import LiteLlm
from dotenv import load_dotenv

load_dotenv()

def create_coordinator_agent():
    # Initialize all sub-agents
    from agents.content_uploader.agent import create_agent as create_content_uploader
    from agents.explainer.agent import create_agent as create_explainer
    from agents.quiz_generator.agent import create_agent as create_quiz_generator
    from agents.flashcard.agent import create_agent as create_flashcard_agent
    from agents.note_formatter.agent import create_agent as create_note_formatter

    return Agent(
        name="ai_tutor_coordinator",
        description="Orchestrates all tutoring activities",
        model=LiteLlm(model="gemini-1.5-pro-latest", api_key=os.getenv("GOOGLE_API_KEY")),
        instruction=(
            "Route user requests to appropriate sub-agents:\n"
            "1. File uploads → ContentUploader\n"
            "2. Explanations → Explainer\n"
            "3. Quizzes → QuizGenerator\n"
            "4. Flashcards → FlashcardAgent\n"
            "5. Notes → NoteFormatter"
        ),
        sub_agents=[
            create_content_uploader(),
            create_explainer(),
            create_quiz_generator(),
            create_flashcard_agent(),
            create_note_formatter()
        ]
    )