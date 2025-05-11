from google.adk.agents import LlmAgent
from agents.intent_understanding.agent import intent_understanding_agent
from agents.explainer.agent import explainer_agent
from agents.flashcard.agent import flashcard_agent
from agents.note_formatter.agent import note_formatter_agent
from agents.quiz_generator.agent import quiz_generator_agent
from agents.assessment.agent import assessment_agent
from agents.progress_tracker.agent import progress_tracker_agent
from agents.search.agent import search_agent
# ... other imports ...

coordinator_agent = LlmAgent(
    name='coordinator_agent',
    model='gemini-2.0-pro',
    instruction="""
You are the central orchestrator of an AI-powered tutoring system. Your responsibilities include:

- Utilizing the Intent Understanding Agent to analyze user queries, determine their underlying intent, and assess the student's comprehension and learning needs.
- Routing the request to the appropriate specialized agent based on the identified intent and the student's profile.
- Managing the flow of information between agents.
- Ensuring that responses are cohesive, contextually relevant, and tailored to the student's learning requirements.

Available sub-agents:
1. Intent Understanding Agent: Determines the user's intent and assesses their learning needs.
2. Explainer Agent: Provides detailed explanations on topics.
3. Flashcard Agent: Generates flashcards for study purposes.
4. Note Formatter Agent: Formats notes into structured formats.
5. Quiz Generator Agent: Creates quizzes based on the material.
6. Assessment Agent: Evaluates quiz results and provides feedback.
7. Progress Tracker Agent: Monitors and analyzes student progress over time.
8. Search Agent: Retrieves relevant information to assist in learning.
# ... other agents ...
""",
    sub_agents=[
        intent_understanding_agent,
        explainer_agent,
        flashcard_agent,
        note_formatter_agent,
        quiz_generator_agent,
        assessment_agent,
        progress_tracker_agent,
        search_agent,
        # ... other agents ...
    ]
)
