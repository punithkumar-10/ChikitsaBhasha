import streamlit as st
import asyncio
from agents.coordinator import create_coordinator_agent
from dotenv import load_dotenv

load_dotenv()

async def main():
    st.set_page_config(page_title="AI Tutor", layout="centered")
    st.title("🤖 AI Tutor")
    
    coordinator = create_coordinator_agent()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "quiz" not in st.session_state:
        st.session_state.quiz = None
    
    # File upload
    uploaded_file = st.file_uploader("Upload learning materials", type=['pdf', 'pptx', 'docx', 'txt'])
    if uploaded_file:
        with st.spinner("Processing your materials..."):
            response = await coordinator.sub_agents[0].run(f"Process {uploaded_file.name}")
            st.session_state.messages.append({"role": "assistant", "content": f"📚 Processed: {uploaded_file.name}"})
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Special handling for quizzes
            if message.get("quiz"):
                st.session_state.quiz = message["quiz"]
                for i, q in enumerate(message["quiz"]["questions"], 1):
                    st.write(f"**Q{i}. {q['question']}**")
                    for opt in q['options']:
                        st.write(opt)
                    st.write(f"*Answer: {q['answer']}*")
                    st.divider()

    if prompt := st.chat_input("Ask or request quizzes/flashcards/notes"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Generating response..."):
            response = await coordinator.run(prompt)
            
            # Special handling for different response types
            if "quiz" in response.lower():
                quiz_data = parse_quiz_response(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Here's your quiz:",
                    "quiz": quiz_data
                })
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})

def parse_quiz_response(text):
    """Convert quiz text response to structured format"""
    questions = []
    current_q = {}
    
    for line in text.split('\n'):
        if line.startswith('Q') and '.' in line:
            if current_q:
                questions.append(current_q)
            current_q = {"question": line.split('. ')[1], "options": []}
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            current_q["options"].append(line)
        elif line.startswith("Answer:"):
            current_q["answer"] = line.split(": ")[1]
    
    if current_q:
        questions.append(current_q)
    
    return {"questions": questions}

if __name__ == "__main__":
    asyncio.run(main())