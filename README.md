# ChikitsaBhasha — Medical Insights in Your Language 🇮🇳

**ChikitsaBhasha** is an AI-powered multilingual medical assistant that explains complex medical reports in **simple, understandable language** — in your **preferred Indian language**. Built using **RAG (Retrieval-Augmented Generation)** and **Google Gemini**, this project aims to make healthcare communication more **inclusive, accessible, and personalized**.

---

## ✨ Features

* 📄 **PDF Medical Report Reader**
  Upload medical reports (digital or scanned) and extract content using OCR + NLP.

* 🔍 **RAG-Based Reasoning**
  Uses sentence-transformer embeddings + Pinecone vector DB for relevant context retrieval.

* 🧠 **Gemini API Integration**
  Generates clear, medically-informed explanations using Google’s Gemini 2.5 model.

* 🗣️ **Multilingual Support**
  Supports 12 Indian languages like Hindi, Tamil, Telugu, Kannada, Bengali, Marathi, and more.

* 💬 **Interactive Chat Interface**
  Ask questions about your medical report and get human-friendly explanations.

---

## 🧠 Multi-Agent Architecture (Powered by Agno)

ChikitsaBhasha now uses the [Agno](https://github.com/agno-agi/agno) multi-agent framework. Each supported language (English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu) has its own dedicated AI agent, all orchestrated by a language router team. All agents use **Google Gemini** as the LLM backend.

- **Language Router:** Automatically detects the user's language and routes the query to the correct agent.
- **Language Agents:** Each agent is responsible for answering only in its assigned language, ensuring accurate and culturally relevant responses.
- **Team Coordination:** The Agno Team routes, coordinates, and aggregates responses, providing a seamless multilingual experience.

---

## 🧪 Example Use Cases

> Upload a medical report and ask:

* “Explain this in Telugu.”
* “What does high creatinine mean?”
* “Is this report normal or dangerous?”
* “Translate this in Marathi for my grandparents.”

---

## ⚙️ Updated Project Flow

1. **Upload a PDF Medical Report**
   The user uploads a scanned or digital PDF. If text cannot be extracted, OCR is used as fallback.

2. **Text Extraction and Chunking**
   The PDF is parsed, and its content is split into overlapping text chunks.

3. **Embedding Creation**
   Each chunk is embedded using a Sentence Transformer (MiniLM) and stored in Pinecone.

4. **User Asks a Question (in any supported language)**
   The user types a question about the report in any Indian language or lets the system auto-detect.

5. **Language Router (Agno Team) detects language and dispatches to the correct agent**
   The user's question is embedded and matched against the most relevant chunks in Pinecone.

6. **Relevant context is retrieved from Pinecone**

7. **Gemini LLM (via Agno) generates a response in the correct language**

8. **Response is delivered via the chat interface**

> 🔹 Built with empathy for patients across linguistic borders.

---

## 🚀 Getting Started

### 🛠 Prerequisites

* Python 3.10+
* Tesseract OCR installed
* Environment variables in `.env`:

  * `GOOGLE_API_KEY`
  * `PINECONE_API_KEY`

### 📦 Installation

```bash
git clone https://github.com/punithkumar-10/chikitsabhasha.git
cd chikitsabhasha
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
python app.py
```

Visit `http://localhost:7860` in your browser.

---

## 🏗️ Tech Stack

- **Agno**: Multi-agent orchestration and language routing
- **Google Gemini**: LLM for all agents
- **Gradio**: Chat UI
- **Pinecone**: Vector database for semantic search
- **Sentence Transformers**: Embedding generation
- **PyPDF2, pdf2image, pytesseract**: PDF and OCR processing

---

## 🆕 How to Use (Quick Start)

1. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
2. Set your API keys in `.env` (see example in repo)
3. Run the app:
   ```bash
   python app.py
   ```
4. Upload a medical report and chat in your preferred language!

---

## 📝 Notes
- All language agents use Gemini via Agno (no OpenAI LLMs required)
- The system is fully modular: add more languages by defining new agents in the code
- For more on Agno, see [Agno Docs](https://github.com/agno-agi/agno-docs)

---

## 📬 Author

**N Punith Kumar**
📧 Email: [punithkumarnimmala@gmail.com](mailto:punithkumarnimmala@gmail.com)
🔗 GitHub: [@punithkumar-10](https://github.com/punithkumar-10)

---

## 🙏 Acknowledgements

* [Google Generative AI](https://ai.google.dev/)
* [Pinecone Vector DB](https://www.pinecone.io/)
* [Gradio UI](https://www.gradio.app/)
* [Tesseract OCR](https://github.com/tesseract-ocr)

---

## 🌐 Supported Languages

* English (en)
* Hindi (hi)
* Tamil (ta)
* Telugu (te)
* Bengali (bn)
* Marathi (mr)
* Gujarati (gu)
* Kannada (kn)
* Malayalam (ml)
* Punjabi (pa)
* Oriya (or)
* Urdu (ur)
