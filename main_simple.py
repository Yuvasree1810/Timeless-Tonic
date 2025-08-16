# NutriMed Bot API - Simplified Version (Windows Compatible)
# This version avoids problematic dependencies like sentence-transformers and chromadb
# To run: python main_simple.py
# API docs: http://localhost:8080/docs
#
# Author: [Your Name]
# --------------------------------------
import os
import re
import json
import hashlib
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import random
import PyPDF2  # PyPDF2 for PDF processing (Windows compatible)
import docx
import uuid
from datetime import datetime
import requests
from fastapi.staticfiles import StaticFiles
import io

# Try to import Groq, but handle gracefully if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Groq not available: {e}")
    GROQ_AVAILABLE = False
    Groq = None

# --- Configuration ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Initialize Components ---
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("⚠️ GROQ_API_KEY not set!")
    print("📝 To set the API key, run one of these commands:")
    print("   Windows: set GROQ_API_KEY=your_api_key_here")
    print("   Linux/Mac: export GROQ_API_KEY=your_api_key_here")
    print("   Or create a .env file with: GROQ_API_KEY=your_api_key_here")
    print("\n🔗 Get your API key from: https://console.groq.com/")
    print("⚠️ Server will start but chat functionality will be limited.")
    api_key = "dummy_key"  # Allow server to start

# Initialize Groq client
client = None
if GROQ_AVAILABLE:
    try:
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully!")
    except Exception as e:
        print(f"⚠️ Groq client initialization failed: {e}")
        print("📝 Chat functionality will be limited.")
        client = None
else:
    print("⚠️ Groq not available. Chat functionality will be limited.")

# Simple in-memory document storage
documents = {}

# --- FastAPI Setup ---
app = FastAPI(title="NutriMed Bot Simple", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/static-root", StaticFiles(directory="."), name="static-root")

# --- Document Processing Functions ---
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from various file formats."""
    if file.filename.endswith('.txt'):
        return file.file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        try:
            pdf_bytes = file.file.read()
            with PyPDF2.PdfReader(io.BytesIO(pdf_bytes)) as doc:
                text = "\n".join(page.extract_text() for page in doc.pages)
            return text
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
            return f"PDF processing failed: {str(e)}. Please try uploading a text file instead."
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file.file)
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError('Unsupported file type. Only .txt, .pdf, and .docx are allowed.')

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better chunking."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}]', ' ', text)
    return text.strip()

def simple_search(query: str, document_chunks: List[str], top_k: int = 3) -> List[str]:
    """Simple keyword-based search."""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in document_chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    
    # Sort by score (descending) and return top chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k]]

# --- Message Classification ---
def classify_message(message: str) -> str:
    """Classify user message to determine response type."""
    msg = message.lower().strip()

    greetings = [
        "hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you",
        "greetings", "yo", "sup", "howdy", "good night", "best wishes", "congratulations", "congrats"
    ]
    if any(greet in msg for greet in greetings):
        if "good morning" in msg:
            return "good_morning"
        if "good night" in msg:
            return "good_night"
        return "greeting"

    # Check for document-related queries
    doc_keywords = ["summarize", "explain", "tell me about", "what does it say", "document", "file"]
    if any(keyword in msg for keyword in doc_keywords):
        return "document_query"

    # Medical queries - expanded with more comprehensive terms
    medical_keywords = [
        "symptom", "headache", "head ache", "head pain", "fever", "pain", "ache", "sore", "hurt",
        "treatment", "medicine", "medication", "pill", "drug", "nutrition", "diet", "food",
        "injury", "hurt", "wound", "cut", "bruise", "cough", "cold", "flu", "infection",
        "rash", "itch", "burn", "doctor", "physician", "nurse", "disease", "illness", "sick",
        "health", "hospital", "clinic", "diabetes", "pressure", "blood pressure", "cholesterol",
        "therapy", "mental", "depression", "anxiety", "stress", "mood", "wellness", "fitness",
        "surgery", "operation", "x-ray", "scan", "mri", "ct", "checkup", "exam", "test",
        "period", "periods", "cramp", "cramps", "menstruation", "menstrual", "bleeding",
        "cycle", "PMS", "dysmenorrhea", "ovulation", "pregnancy", "fertility", "drug", 
        "side effect", "nausea", "vomit", "dizzy", "dizziness", "weak", "weakness", "tired",
        "fatigue", "sleep", "insomnia", "stomach", "stomachache", "stomach pain", "belly",
        "chest", "chest pain", "heart", "heart pain", "back", "back pain", "joint", "joint pain",
        "muscle", "muscle pain", "bone", "bone pain", "throat", "sore throat", "ear", "ear pain",
        "eye", "eye pain", "vision", "blur", "blurry", "nose", "runny nose", "congestion",
        "breathing", "breath", "shortness", "wheezing", "skin", "acne", "pimple", "wart",
        "mole", "swelling", "swollen", "lump", "bump", "tumor", "cancer", "tumor",
        "nose", "runny", "running", "snot", "nasal", "congestion", "stuffy", "sneeze",
        "sneezing", "allergy", "allergic", "cold", "flu", "virus", "infection"
    ]
    if any(word in msg for word in medical_keywords):
        return "medical"

    return "irrelevant"

# --- API Endpoints ---
@app.post('/upload')
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for simple storage."""
    try:
        # Extract text
        text = extract_text_from_file(file)
        
        # Preprocess and chunk text
        text = preprocess_text(text)
        chunks = chunk_text(text)
        
        if not chunks:
            raise ValueError("No valid chunks extracted from document")
        
        # Store document in memory
        doc_id = str(uuid.uuid4())
        documents[doc_id] = {
            'filename': file.filename,
            'chunks': chunks,
            'upload_time': datetime.now().isoformat(),
            'text': text
        }
        
        # Generate summary if Groq is available
        summary = "Document uploaded successfully. Summary not available without Groq API."
        if client:
            try:
                summary_prompt = f"""You are a medical assistant. Analyze the following document and provide a clear, structured summary.

Document content:
{text[:2000]}...

Please provide a summary that includes:
1. Document type (medical report, prescription, lab results, etc.)
2. Key findings or diagnoses
3. Important medications or treatments mentioned
4. Any critical dates or patient information
5. Recommendations or follow-up instructions

Keep the summary concise but comprehensive."""

                chat_history = [
                    {"role": "system", "content": "You are a medical assistant. Provide clear, accurate summaries of medical documents."},
                    {"role": "user", "content": summary_prompt}
                ]
                
                completion = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=chat_history,
                    temperature=0.3,
                    max_tokens=400
                )
                summary = completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"Summary generation failed: {e}")
                summary = "Document uploaded successfully. Summary generation failed."
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "summary": summary,
            "message": f"Document '{file.filename}' uploaded and processed successfully!"
        }
        
    except Exception as e:
        print(f"[ERROR] Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint with medical responses and document search."""
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        doc_id = data.get("doc_id")

        if not user_message:
            return JSONResponse({"response": "⚠️ Please enter a message."}, status_code=400)

        # If doc_id is provided, search in the document
        if doc_id and doc_id in documents:
            doc_data = documents[doc_id]
            relevant_chunks = simple_search(user_message, doc_data['chunks'], top_k=3)
            
            if relevant_chunks:
                context_text = "\n".join(relevant_chunks)
                
                if client:
                    try:
                        system_prompt = (
                            "You are a medical assistant chatbot. Answer questions using ONLY the information from the provided document. "
                            "CRITICAL RULES:\n"
                            "1. Answer questions using ONLY information from the provided document\n"
                            "2. If the document doesn't contain relevant information, respond with: 'The document does not contain enough information to answer that question.'\n"
                            "3. Do NOT use any outside knowledge, medical expertise, or assumptions\n"
                            "4. Stay grounded strictly in the content of the uploaded document\n"
                            "5. Be specific and reference exact details from the document\n\n"
                            "Document Content:\n" + context_text
                        )
                        
                        chat_history = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ]
                        
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=chat_history,
                            temperature=0.3,
                            max_completion_tokens=1024,
                            top_p=0.9,
                            stream=False,
                            stop=None
                        )
                        
                        response_text = completion.choices[0].message.content.strip()
                        return {"response": response_text}
                    except Exception as e:
                        print(f"Groq API error: {e}")
                        return {"response": "The document does not contain enough information to answer that question."}
                else:
                    return {"response": "Document uploaded successfully. Please set up your GROQ_API_KEY to enable chat functionality."}
            else:
                return {"response": "The document does not contain enough information to answer that question."}
        else:
            # Classify the message
            msg_type = classify_message(user_message)
            print(f"[DEBUG] Message: '{user_message}' classified as: {msg_type}")

            if msg_type == "greeting":
                return {"response": random.choice([
                    "👋 Hello! How can I assist you with your health or wellness today?",
                    "Hi there! 😊 What medical or nutrition question can I help with?",
                    "Hey! I'm here to help with symptoms, medicines, and more. Ask away!"
                ])}
            elif msg_type == "good_morning":
                return {"response": "🌞 Good morning! How can I help you feel your best today?"}
            elif msg_type == "good_night":
                return {"response": "🌙 Good night! If you have any health questions before bed, let me know."}
            elif msg_type == "medical":
                # Provide helpful medical responses for general health questions
                if client:
                    try:
                        medical_prompt = f"""You are a helpful medical assistant. The user is asking about their health: "{user_message}"

Please provide a helpful, informative response that includes:
1. General information about the condition/symptom
2. Common causes
3. When to seek medical attention
4. General lifestyle tips or home remedies (if appropriate)
5. A reminder that this is not a substitute for professional medical advice

Keep your response friendly, informative, and reassuring. Always encourage consulting a healthcare professional for proper diagnosis and treatment."""

                        chat_history = [
                            {"role": "system", "content": "You are a helpful medical assistant. Provide informative and reassuring responses to health questions while always encouraging professional medical consultation."},
                            {"role": "user", "content": medical_prompt}
                        ]
                        
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=chat_history,
                            temperature=0.7,
                            max_completion_tokens=800,
                            top_p=0.9,
                            stream=False,
                            stop=None
                        )
                        
                        response_text = completion.choices[0].message.content.strip()
                        return {"response": response_text}
                    except Exception as e:
                        print(f"Groq API error: {e}")
                        # Fallback responses for common medical questions
                        return get_fallback_medical_response(user_message)
                else:
                    return get_fallback_medical_response(user_message)
            elif msg_type == "document_query":
                return {"response": "📄 Please upload a document first, then ask your question. I can only answer questions based on the content of your uploaded document."}
            else:
                return {"response": "🤖 I'm here to help with medical questions! Feel free to ask about symptoms, treatments, or general health concerns."}

    except Exception as e:
        print(f"[ERROR] /chat endpoint failed: {e}")
        return {"response": "Sorry, there was an error processing your request. Please try again later."}

def get_fallback_medical_response(user_message: str) -> str:
    """Provide fallback medical responses when Groq is not available."""
    msg = user_message.lower()
    
    # Common medical responses
    if "headache" in msg or "head pain" in msg:
        return """🤕 Headaches can be caused by various factors like stress, dehydration, lack of sleep, or eye strain.

Common causes:
• Tension headaches (most common)
• Migraines
• Sinus problems
• Dehydration
• Poor posture

Try these remedies:
• Rest in a quiet, dark room
• Stay hydrated
• Apply a cold or warm compress
• Practice relaxation techniques
• Take over-the-counter pain relievers

⚠️ Seek medical attention if:
• Headache is severe or sudden
• Accompanied by fever, confusion, or vision changes
• Headache after head injury
• Persistent headaches

Remember: This is general information. For proper diagnosis, please consult a healthcare professional."""
    
    elif "stomach" in msg and "pain" in msg or "stomachache" in msg:
        return """🤢 Stomach pain can have many causes, from mild to serious.

Common causes:
• Indigestion or gas
• Food poisoning
• Stomach flu
• Acid reflux
• Stress or anxiety

Try these remedies:
• Rest and avoid solid foods initially
• Sip clear fluids (water, broth)
• Apply gentle heat to abdomen
• Avoid spicy, fatty, or acidic foods
• Take over-the-counter antacids

⚠️ Seek medical attention if:
• Severe or persistent pain
• Pain with fever or vomiting
• Blood in stool
• Pain that spreads to chest or back
• Pain after eating

Remember: This is general information. For proper diagnosis, please consult a healthcare professional."""
    
    elif "fever" in msg:
        return """🌡️ Fever is your body's natural response to infection or illness.

Common causes:
• Viral infections (cold, flu)
• Bacterial infections
• Inflammatory conditions
• Heat exhaustion

Home care:
• Rest and stay hydrated
• Take acetaminophen or ibuprofen
• Use cool compresses
• Wear light clothing
• Monitor temperature

⚠️ Seek medical attention if:
• Temperature above 103°F (39.4°C)
• Fever lasting more than 3 days
• Fever with severe headache or rash
• Fever in infants under 3 months
• Fever with confusion or seizures

Remember: This is general information. For proper diagnosis, please consult a healthcare professional."""
    
    elif "cough" in msg:
        return """🤧 Coughing helps clear your airways and is often a symptom of other conditions.

Common causes:
• Upper respiratory infections
• Allergies
• Acid reflux
• Asthma
• Post-nasal drip

Home remedies:
• Stay hydrated
• Use honey (for adults)
• Humidify the air
• Rest your voice
• Avoid irritants (smoke, dust)

⚠️ Seek medical attention if:
• Cough lasting more than 2 weeks
• Cough with blood or thick mucus
• Cough with chest pain or difficulty breathing
• Cough with fever
• Cough in infants

Remember: This is general information. For proper diagnosis, please consult a healthcare professional."""
    
    elif "pain" in msg:
        return """😣 Pain is your body's way of signaling that something needs attention.

General pain management:
• Rest the affected area
• Apply ice or heat as appropriate
• Take over-the-counter pain relievers
• Practice gentle stretching
• Use proper posture

⚠️ Seek medical attention if:
• Severe or sudden pain
• Pain that doesn't improve
• Pain with other symptoms (fever, swelling)
• Pain after injury
• Chronic pain affecting daily life

Remember: This is general information. For proper diagnosis, please consult a healthcare professional."""
    
    else:
        return """🤖 I understand you're experiencing health concerns. While I can provide general information, it's important to remember that I'm not a substitute for professional medical advice.

For your specific situation, I recommend:
• Consulting with a healthcare professional
• Describing your symptoms in detail
• Following up on any concerning symptoms
• Keeping track of when symptoms occur

If you have a medical document you'd like me to analyze, feel free to upload it and I can help answer questions about its content.

Take care and stay healthy! 💙"""

@app.get("/")
def get_homepage():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    try:
        doc_list = []
        for doc_id, doc_data in documents.items():
            doc_list.append({
                'doc_id': doc_id,
                'filename': doc_data['filename'],
                'upload_time': doc_data['upload_time'],
                'chunks': len(doc_data['chunks'])
            })
        return {"documents": doc_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document."""
    try:
        if doc_id in documents:
            del documents[doc_id]
            return {"message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NutriMed Bot (Simplified Version)...")
    print("📝 This version uses simple in-memory storage and keyword search.")
    print("🔗 API docs: http://localhost:8080/docs")
    print("🌐 Web interface: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
