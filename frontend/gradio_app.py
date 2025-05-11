import gradio as gr
import os
from agents.coordinator.agent import coordinator_agent
import PyPDF2

# Global variable to store uploaded content
uploaded_content = ""

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        text = f"Error reading PDF: {e}"
    return text

def chat_function(message, history):
    global uploaded_content
    if isinstance(message, dict):
        # Handle file uploads
        files = message.get("files", [])
        texts = []
        for file_path in files:
            if file_path.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
                texts.append(text)
            else:
                texts.append(f"Unsupported file type: {file_path}")
        uploaded_content += "\n".join(texts)
        return "Files uploaded and content extracted successfully."
    else:
        # Handle text messages
        prompt = f"{uploaded_content}\n\nUser: {message}"
        response = coordinator_agent.run(prompt)
        return response

demo = gr.ChatInterface(
    fn=chat_function,
    title="AI Tutor",
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_types=[".pdf"],
        file_count="multiple",
        label="Type your message or upload PDFs"
    )
)

if __name__ == "__main__":
    demo.launch()
