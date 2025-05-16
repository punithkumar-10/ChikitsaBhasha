import gradio as gr
import google.generativeai as genai
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import hashlib
import uuid
import langdetect
import os
from io import BytesIO
from datetime import datetime
import time
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests
from dotenv import load_dotenv


load_dotenv()
# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Document Processor
class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_cache = {}

    def extract_text_from_pdf(self, pdf_file):
        try:
            # Handle Gradio's NamedString, UploadedFile, or plain str/bytes
            file_bytes = None
            if hasattr(pdf_file, 'file'):
                file_obj = pdf_file.file
                file_obj.seek(0)
                file_bytes = file_obj.read()
            elif hasattr(pdf_file, 'seek') and hasattr(pdf_file, 'read'):
                pdf_file.seek(0)
                file_bytes = pdf_file.read()
            elif isinstance(pdf_file, (bytes, bytearray)):
                file_bytes = pdf_file
            elif isinstance(pdf_file, str):
                # If it's a path, read the file
                with open(pdf_file, 'rb') as f:
                    file_bytes = f.read()
            else:
                raise ValueError("Unsupported file type for PDF extraction.")
            file_hash = hashlib.md5(file_bytes).hexdigest()
            if file_hash in self.memory_cache:
                print("[PDF] Using cached version.")
                return self.memory_cache[file_hash]
            pdf_bytes = BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            text = ""
            print(f"[PDF] Number of pages: {len(pdf_reader.pages)}")
            for i, page in enumerate(pdf_reader.pages):
                page_text = None
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    print(f"Error extracting text from page {i+1}: {str(e)}")
                if not page_text and hasattr(page, 'extract_text'):
                    try:
                        page_text = page.extract_text(layout_mode='raw')
                    except Exception as e:
                        print(f"Alternative extraction failed on page {i+1}: {str(e)}")
                if page_text:
                    print(f"[PDF] Extracted {len(page_text)} chars from page {i+1}")
                    text += page_text
                else:
                    print(f"[PDF] No text extracted from page {i+1}")
            if not text.strip():
                print("[PDF] No text extracted using PyPDF2, attempting OCR...")
                try:
                    pdf_bytes.seek(0)
                    images = convert_from_bytes(pdf_bytes.read())
                    for i, image in enumerate(images):
                        try:
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text:
                                print(f"[OCR] Extracted {len(ocr_text)} chars from page {i+1}")
                                text += ocr_text
                            else:
                                print(f"[OCR] No text found on page {i+1}")
                        except Exception as e:
                            print(f"Error performing OCR on page {i+1}: {str(e)}")
                except Exception as e:
                    print(f"OCR extraction failed: {str(e)}")
            self.memory_cache[file_hash] = text
            print(f"[PDF] Total extracted text length: {len(text)}")
            return text
        except Exception as e:
            print(f"PDF extraction failed: {str(e)}")
            return ""

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def create_embeddings(self, texts):
        embeddings = self.model.encode(texts).tolist()
        return embeddings

# Pinecone Handler
class PineconeHandler:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = "rag-documents"
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                # Wait for index to be ready
                attempts = 0
                while attempts < 10:
                    time.sleep(2)
                    if self.index_name in [idx.name for idx in self.pc.list_indexes()]:
                        break
                    attempts += 1
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            print(f"Error setting up Pinecone: {str(e)}")
            # Create a dummy index for testing if Pinecone fails
            self.index = None

    def upsert_vectors(self, vectors, texts, file_name, doc_id=None):
        if self.index is None:
            print("Pinecone index not available. Vector storage skipped.")
            return
        try:
            vectors_to_upsert = [
                {
                    "id": f"{file_name}_{i}",
                    "values": vector,
                    "metadata": {
                        "text": text,
                        "source": file_name,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                for i, (vector, text) in enumerate(zip(vectors, texts))
            ]
            self.index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"Error upserting vectors: {str(e)}")

    def query_vectors(self, query_vector, top_k=3):
        if self.index is None:
            return type('obj', (object,), {'matches': []})
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            print(f"Error querying vectors: {str(e)}")
            return type('obj', (object,), {'matches': []})

# Gemini Handler
class GeminiHandler:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        self.chat_sessions = {}

    def get_response(self, prompt, session_id=None):
        try:
            if session_id and session_id in self.chat_sessions:
                chat = self.chat_sessions[session_id]
                response = chat.send_message(prompt)
            else:
                chat = self.model.start_chat(history=[])
                response = chat.send_message(prompt)
                if session_id:
                    self.chat_sessions[session_id] = chat
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

def translate_text(text, target_lang):
    """
    Translates text to the target language using Google Translate.
    This function ensures the translation works for all supported languages.
    """
    if target_lang == 'en' or not text.strip():
        return text
    
    # If auto, try to detect the language
    if target_lang == 'auto':
        try:
            target_lang = langdetect.detect(text)
            if target_lang not in ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'ur']:
                target_lang = 'en'
        except Exception:
            target_lang = 'en'
    
    try:
        print(f"[Translation] Translating to {target_lang}")
        url = "https://translate.googleapis.com/translate_a/single"
        
        # First try direct translation from detected language to target
        params = {
            'client': 'gtx',
            'sl': 'auto',  # Let Google detect the source language
            'tl': target_lang,
            'dt': 't',
            'q': text
        }
        
        response = requests.get(url, params=params, timeout=10)  # Add timeout
        if response.status_code == 200:
            result = response.json()
            translated = ''.join([item[0] for item in result[0]])
            print(f"[Translation] Successfully translated to {target_lang}")
            return translated
        else:
            print(f"[Translation] Error: {response.status_code}")
            # Fallback to English first if direct translation failed
            return text + f"\n\n[Translation unavailable for {target_lang}]"
    except Exception as e:
        print(f"[Translation] Exception: {str(e)}")
        return text + f"\n\n[Translation error: {str(e)}]"

# MultiLanguage Agent
class MultiLanguageAgent:
    def __init__(self, gemini_handler):
        self.gemini = gemini_handler
        self.language_prompts = {
            'en': "You are a medical report assistant. Provide clear, helpful, and accurate explanations of the uploaded medical report in English, using simple language understandable by patients. Always remind users that this is not a substitute for professional medical advice.",
            'hi': "आप एक चिकित्सा रिपोर्ट सहायक हैं। कृपया अपलोड की गई मेडिकल रिपोर्ट की स्पष्ट, सहायक और सटीक व्याख्या सरल हिंदी में दें, ताकि मरीज आसानी से समझ सकें। हमेशा याद दिलाएं कि यह पेशेवर चिकित्सा सलाह का विकल्प नहीं है।",
            'ta': "நீங்கள் ஒரு மருத்துவ அறிக்கை உதவியாளர். பதிவேற்றப்பட்ட மருத்துவ அறிக்கையின் தெளிவான, உதவிகரமான மற்றும் துல்லியமான விளக்கங்களை எளிய தமிழில் வழங்கவும். இது மருத்துவ ஆலோசனையின் மாற்றாக அல்ல என்பதை எப்போதும் நினைவூட்டவும்.",
            'te': "మీరు వైద్య నివేదిక సహాయకుడు. అప్‌లోడ్ చేసిన వైద్య నివేదికను సులభంగా అర్థమయ్యే తెలుగులో స్పష్టంగా, సహాయకంగా, ఖచ్చితంగా వివరించండి. ఇది వైద్య నిపుణుల సలహాకు ప్రత్యామ్నాయం కాదని ఎప్పుడూ గుర్తు చేయండి.",
            'bn': "আপনি একজন চিকিৎসা রিপোর্ট সহকারী। দয়া করে আপলোড করা মেডিকেল রিপোর্টের স্পষ্ট, সহায়ক এবং সঠিক ব্যাখ্যা সহজ বাংলায় দিন, যাতে রোগীরা সহজে বুঝতে পারেন। এটি কখনই পেশাদার চিকিৎসা পরামর্শের বিকল্প নয় তা মনে করিয়ে দিন।",
            'mr': "आपण वैद्यकीय अहवाल सहाय्यक आहात. कृपया अपलोड केलेल्या वैद्यकीय अहवालाचे स्पष्ट, उपयुक्त आणि अचूक स्पष्टीकरण सोप्या मराठीत द्या. हे व्यावसायिक वैद्यकीय सल्ल्याचा पर्याय नाही हे नेहमी लक्षात ठेवा.",
            'gu': "તમે એક મેડિકલ રિપોર્ટ સહાયક છો. કૃપા કરીને અપલોડ કરેલા મેડિકલ રિપોર્ટનું સ્પષ્ટ, ઉપયોગી અને ચોક્કસ સ્પષ્ટીકરણ સરળ ગુજરાતીમાં આપો. હંમેશા યાદ રાખો કે આ વ્યાવસાયિક તબીબી સલાહનો વિકલ્પ નથી.",
            'kn': "ನೀವು ವೈದ್ಯಕೀಯ ವರದಿ ಸಹಾಯಕರು. ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ವೈದ್ಯಕೀಯ ವರದಿಯ ಸ್ಪಷ್ಟ, ಸಹಾಯಕ ಮತ್ತು ಖಚಿತ ವಿವರಣೆಯನ್ನು ಸರಳ ಕನ್ನಡದಲ್ಲಿ ನೀಡಿ. ಇದು ವೃತ್ತಿಪರ ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ ಎಂಬುದನ್ನು ಯಾವಾಗಲೂ ನೆನಪಿಸಿ.",
            'ml': "നിങ്ങൾ ഒരു മെഡിക്കൽ റിപ്പോർട്ട് അസിസ്റ്റന്റാണ്. അപ്‌ലോഡ് ചെയ്ത മെഡിക്കൽ റിപ്പോർട്ടിന്റെ വ്യക്തമായ, സഹായകരമായ, കൃത്യമായ വിശദീകരണം ലളിതമായ മലയാളത്തിൽ നൽകുക. ഇത് പ്രൊഫഷണൽ മെഡിക്കൽ ഉപദേശത്തിന് പകരം അല്ലെന്ന് എപ്പോഴും ഓർമ്മപ്പെടുത്തുക.",
            'pa': "ਤੁਸੀਂ ਇੱਕ ਮੈਡੀਕਲ ਰਿਪੋਰਟ ਸਹਾਇਕ ਹੋ। ਕਿਰਪਾ ਕਰਕੇ ਅੱਪਲੋਡ ਕੀਤੀ ਮੈਡੀਕਲ ਰਿਪੋਰਟ ਦੀ ਸਾਫ਼, ਮਦਦਗਾਰ ਅਤੇ ਸਹੀ ਵਿਆਖਿਆ ਆਸਾਨ ਪੰਜਾਬੀ ਵਿੱਚ ਦਿਓ। ਹਮੇਸ਼ਾ ਯਾਦ ਦਿਵਾਓ ਕਿ ਇਹ ਪੇਸ਼ੇਵਰ ਤਬੀਬੀ ਸਲਾਹ ਦਾ ਵਿਕਲਪ ਨਹੀਂ ਹੈ।",
            'or': "ଆପଣ ଏକ ଚିକିତ୍ସା ରିପୋର୍ଟ ସହାୟକ। ଦୟାକରି ଅପଲୋଡ୍ କରାଯାଇଥିବା ମେଡିକାଲ୍ ରିପୋର୍ଟର ସ୍ପଷ୍ଟ, ସହାୟକ ଏବଂ ସଠିକ୍ ବ୍ୟାଖ୍ୟା ସହଜ ଓଡ଼ିଆରେ ଦିଅନ୍ତୁ। ଏହା କେବେ ବି ପେଶାଦାର ଚିକିତ୍ସା ପରାମର୍ଶର ବିକଳ୍ପ ନୁହେଁ ବୋଲି ସମୟ ସମୟରେ ମନେ ପକାନ୍ତୁ।",
            'ur': "آپ ایک میڈیکل رپورٹ اسسٹنٹ ہیں۔ براہ کرم اپ لوڈ کی گئی میڈیکل رپورٹ کی واضح، مددگار اور درست وضاحت آسان اردو میں فراہم کریں۔ ہمیشہ یاد دلائیں کہ یہ پیشہ ورانہ طبی مشورے کا متبادل نہیں ہے۔"
        }
        
    def detect_language(self, text):
        """Detect the language of the input text"""
        try:
            detected = langdetect.detect(text)
            return detected if detected in self.language_prompts else 'en'
        except:
            return 'en'
            
    def get_response(self, query, context, language, session_id):
        """Get response from Gemini and ensure it's in the correct language"""
        # If language is auto, detect from query
        if language == "auto":
            language = self.detect_language(query)
            print(f"[Language] Auto-detected language: {language}")
        
        # Always use English for better Gemini comprehension,
        # then translate to target language
        system_prompt = self.language_prompts['en']
        
        # Use a dynamic prompt that ensures the LLM knows we want the response translated
        prompt = f"""
You are a medical report assistant. Provide clear, helpful, and accurate explanations of 
the uploaded medical report in ENGLISH, using simple language understandable by patients.

Context from medical reports:
{context}

User query: {query}

Please respond in English. Your response will be automatically translated to the user's language.
Always remind users that this is not a substitute for professional medical advice.
"""
        # Get the response from Gemini
        response = self.gemini.get_response(prompt, session_id)
        
        # Always translate if language is not English and not auto
        if language != 'en':
            print(f"[Language] Translating response to {language}")
            translated_response = translate_text(response, language)
            return translated_response
        
        return response

# Gradio Interface State
session_id = str(uuid.uuid4())
doc_processor = DocumentProcessor()
pinecone_handler = PineconeHandler()
gemini_handler = GeminiHandler()
multi_agent = MultiLanguageAgent(gemini_handler)

# Store processed reports in memory for this session
processed_docs = {}

# Supported languages for chat
LANGUAGES = [
    ("English / अंग्रेज़ी", "en"),
    ("हिन्दी", "hi"),
    ("தமிழ்", "ta"),
    ("తెలుగు", "te"),
    ("বাংলা", "bn"),
    ("मराठी", "mr"),
    ("ગુજરાતી", "gu"),
    ("ಕನ್ನಡ", "kn"),
    ("മലയാളം", "ml"),
    ("ਪੰਜਾਬੀ", "pa"),
    ("ଓଡ଼ିଆ", "or"),
    ("اردو", "ur")
]

# Store selected language in session
selected_language = gr.State("auto")

def process_pdf_ui(pdf_file):
    if pdf_file is None:
        return "No file uploaded.", "No medical reports processed yet."
    try:
        if not pdf_file.name.lower().endswith('.pdf'):
            return "⚠️ Please upload a PDF file.", "No medical reports processed yet."
        text = doc_processor.extract_text_from_pdf(pdf_file)
        if not text or len(text.strip()) == 0:
            return "⚠️ Could not extract any text from the PDF. It might be a scanned report or image-based PDF.", "No medical reports processed yet."
        chunks = doc_processor.chunk_text(text)
        if not chunks:
            return "⚠️ Could not create text chunks from the report.", "No medical reports processed yet."
        embeddings = doc_processor.create_embeddings(chunks)
        
        # Clear previous context for every new doc
        processed_docs.clear()
        
        # Delete all previous vectors from Pinecone to ensure clean context
        if pinecone_handler.index is not None:
            try:
                # Delete all existing vectors
                pinecone_handler.index.delete(delete_all=True)
                print("[Pinecone] Successfully deleted all previous vectors")
            except Exception as e:
                print(f"[Pinecone] Error when deleting previous vectors: {str(e)}")
        
        # Add new vectors for the current document
        pinecone_handler.upsert_vectors(embeddings, chunks, pdf_file.name, doc_id=None)
        processed_docs[pdf_file.name] = {
            'chunks': len(chunks),
            'text_length': len(text),
            'processed_at': datetime.now().isoformat(),
        }
        return (
            f"✅ Successfully processed document.\n- {len(text):,} characters extracted\n- {len(chunks)} chunks created", 
            ""
        )
    except Exception as e:
        error_message = str(e)
        return f"❌ Error processing PDF: {error_message}", "No medical reports processed successfully."

def predict(message, history, chat_language):
    """Process the user's message and generate a response"""
    # Check if any document has been processed
    if not processed_docs:
        return "Please upload and process a medical report first."
    
    # Always use the language selected in the radio button, unless it's set to auto
    if chat_language and chat_language != "auto":
        language = chat_language
        print(f"[Language] Using selected language: {language}")
    else:
        language = multi_agent.detect_language(message)
        print(f"[Language] Auto-detected language: {language}")
    
    # Get the latest document
    if processed_docs:
        last_doc = list(processed_docs.keys())[-1]
        print(f"[Document] Using document: {last_doc}")
    
    # Embed the query and search for relevant context
    query_embedding = doc_processor.create_embeddings([message])[0]
    results = pinecone_handler.query_vectors(query_embedding)
    context = ""
    if results.matches:
        context = "\n".join([match.metadata['text'] for match in results.matches])
        print(f"[Context] Found {len(results.matches)} relevant chunks")
    
    # Generate response in the selected language
    print(f"[Response] Generating response in {language}")
    response = multi_agent.get_response(message, context, language, session_id)
    
    # Add language indicator to the response for UI display
    lang_names = {
        'en': 'English', 'hi': 'हिन्दी', 'ta': 'தமிழ்', 'te': 'తెలుగు', 
        'bn': 'বাংলা', 'mr': 'मराठी', 'gu': 'ગુજરાતી', 'kn': 'ಕನ್ನಡ',
        'ml': 'മലയാളം', 'pa': 'ਪੰਜਾਬੀ', 'or': 'ଓଡ଼ିଆ', 'ur': 'اردو',
        'auto': 'Auto-detected'
    }
    
    lang_display = lang_names.get(language, language)
    response_with_lang = f"{response}\n\n*Language: {lang_display}*"
    
    return response_with_lang

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), fill_height=True, css="""
body { background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%); }
.gradio-container { background: transparent !important; }
#main-card { box-shadow: 0 8px 32px 0 rgba(60,60,60,0.12); border-radius: 22px; background: var(--gr-secondary-100, #fff); padding: 36px 32px 32px 32px; margin-top: 32px; margin-bottom: 32px; border: 1.5px solid #e0e7ef; }
#left-panel { background: var(--gr-secondary-50, #f4f6fa); border-radius: 16px; box-shadow: 0 2px 8px 0 rgba(60,60,60,0.04); padding: 28px 20px; }
#right-panel { background: var(--gr-secondary-50, #f4f6fa); border-radius: 16px; box-shadow: 0 2px 8px 0 rgba(60,60,60,0.04); padding: 28px 20px; }
#brand-title { font-size: 2.5em; font-weight: 900; letter-spacing: -1px; color: #2563eb; text-align: center; margin-bottom: 0.2em; text-shadow: 0 2px 8px #e0e7ef; }
#brand-desc { font-size: 1.22em; color: #6366f1; text-align: center; margin-bottom: 1.5em; font-weight: 600; }
#footer { font-size: 1.08em; color: #6b7280; text-align: center; margin-top: 2em; letter-spacing: 0.01em; }
#process-status, #docs-status { font-size: 1.12em; }
.gr-button-primary { background: linear-gradient(90deg, #2563eb 60%, #6366f1 100%) !important; color: #fff !important; font-weight: 600; border-radius: 8px !important; box-shadow: 0 2px 8px 0 rgba(60,60,60,0.08); }
.gr-button-primary:hover { background: linear-gradient(90deg, #1d4ed8 60%, #6366f1 100%) !important; }
.gr-radio label { font-size: 1.08em; font-weight: 500; color: #2563eb; }
.gr-file label { font-size: 1.08em; font-weight: 500; color: #6366f1; }
.gr-textbox textarea { font-size: 1.08em; border-radius: 8px; }
@media (max-width: 900px) { #main-card { padding: 16px 2vw; } }
@media (max-width: 700px) { #main-card { flex-direction: column !important; } }
""") as demo:
    gr.Markdown("""
<div id='brand-title'>
  <span style='color:#2563eb;'>Chikitsa</span><span style='color:#6366f1;'>Bhasha</span>
</div>
<div id='brand-desc' style='font-size:1.18em;'>
  Medical Insights in Your Language<br>
</div>
""")
    with gr.Row(elem_id="main-card"):
        with gr.Column(scale=1, elem_id="left-panel"):
            gr.Markdown("<span style='font-size:1.18em;font-weight:700;color:#2563eb;'>Language & Report Upload</span>")
            lang_buttons = gr.Radio(
                choices=[("Auto-detect / स्वतः-चयन", "auto")] + [(name, code) for name, code in LANGUAGES],
                value="auto",
                label="Chat Language",
                interactive=True
            )
            lang_status = gr.Markdown("**Current language**: Auto-detect", elem_id="lang-status")
            
            # Display change in selected language
            def update_lang_status(lang_value):
                lang_names = {
                    'en': 'English', 'hi': 'हिन्दी', 'ta': 'தமிழ்', 'te': 'తెలుగు', 
                    'bn': 'বাংলা', 'mr': 'मराठी', 'gu': 'ગુજરાતી', 'kn': 'ಕನ್ನಡ',
                    'ml': 'മലയാളം', 'pa': 'ਪੰਜਾਬੀ', 'or': 'ଓଡ଼ିଆ', 'ur': 'اردو',
                    'auto': 'Auto-detect'
                }
                lang_name = lang_names.get(lang_value, lang_value)
                return f"**Current language**: {lang_name}"
                
            # Update language status when language selection changes
            lang_buttons.change(
                update_lang_status,
                inputs=lang_buttons,
                outputs=lang_status
            )
            
            gr.Markdown("<span style='font-size:1.12em;font-weight:600;color:#6366f1;'>Upload your medical report (PDF)</span>")
            pdf_input = gr.File(label="Upload Medical Report (PDF)", file_types=[".pdf"])
            process_btn = gr.Button("Process Medical Report", variant="primary")
            process_output = gr.Markdown("", elem_id="process-status")
            docs_output = gr.Markdown("", elem_id="docs-status")
            pdf_input.change(
                process_pdf_ui,
                inputs=pdf_input,
                outputs=[process_output, docs_output],
                queue=False
            )
            process_btn.click(
                process_pdf_ui,
                inputs=pdf_input,
                outputs=[process_output, docs_output],
                queue=False
            )
        with gr.Column(scale=2, elem_id="right-panel"):
            # Redefine chat interface to always use the latest value of the language radio button
            def chat_predict(message, history):
                # Always get the latest value of the language radio button
                try:
                    lang = lang_buttons.value
                except Exception:
                    lang = "auto"
                return predict(message, history, lang)
            chat = gr.ChatInterface(
                fn=chat_predict,
                chatbot=gr.Chatbot(
                    label="Medical Report Chat",
                    type="messages", 
                    show_copy_button=True,
                    render_markdown=True,
                ),
                textbox=gr.Textbox(
                    placeholder="Ask about your medical report in your regional language...",
                    show_label=False,
                    lines=1,
                    interactive=True,
                    submit_btn="Send"
                )
            )
    gr.Markdown("---")
    gr.Markdown("""
<div id='footer' style='text-align:left;'>Minimal RAG Chatbot for Medical Reports in Indian Regional Languages. <span style='float:right'>Built with ❤️ by Punith Kumar</span></div>
""")

demo.launch()
