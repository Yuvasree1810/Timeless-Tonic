# NutriMed Bot API - FastAPI Application
# This file defines the FastAPI backend for NutriMed Bot.
# To run: python main.py
# API docs: http://localhost:8000/docs
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
from groq import Groq
import random
import fitz  # PyMuPDF for PDF
import docx
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime
from ollama_client import OllamaClient
import requests
from fastapi.staticfiles import StaticFiles

# --- RAG Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# --- Initialize Components ---
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not set!")
    print("📝 To set the API key, run one of these commands:")
    print("   Windows: set GROQ_API_KEY=your_api_key_here")
    print("   Linux/Mac: export GROQ_API_KEY=your_api_key_here")
    print("   Or create a .env file with: GROQ_API_KEY=your_api_key_here")
    print("\n🔗 Get your API key from: https://console.groq.com/")
    raise RuntimeError("GROQ_API_KEY not set. Please set the environment variable or create a .env file.")

client = Groq(api_key=api_key)
ollama_client = OllamaClient()

# Initialize embedding model
print("Loading embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    print("🔄 Trying alternative model...")
    try:
        # Fallback to a simpler model
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        print("✅ Alternative embedding model loaded successfully!")
    except Exception as e2:
        print(f"❌ Failed to load any embedding model: {e2}")
        print("⚠️ RAG features will be disabled. Using basic search only.")
        embedding_model = None

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "medical_documents"

try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Medical documents for RAG chatbot"}
    )
    print(f"Created new collection: {collection_name}")

# --- FastAPI Setup ---
app = FastAPI(title="NutriMed Bot RAG", version="2.0")

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
        pdf_bytes = file.file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        return text
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

# --- RAG Functions ---
def add_document_to_vector_db(text: str, filename: str, metadata: Dict = None) -> str:
    """Add document chunks to vector database."""
    # Preprocess text
    text = preprocess_text(text)
    
    # Chunk the text
    chunks = chunk_text(text)
    
    if not chunks:
        raise ValueError("No valid chunks extracted from document")
    
    # Check if embedding model is available
    if embedding_model is None:
        # Fallback: store without embeddings (basic keyword search only)
        doc_id = str(uuid.uuid4())
        chunk_metadata = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata.append({
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_id": doc_id,
                "upload_time": datetime.now().isoformat(),
                **(metadata or {})
            })
            chunk_ids.append(f"{doc_id}_chunk_{i}")
        
        # Add to ChromaDB without embeddings
        collection.add(
            documents=chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
        
        return doc_id
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Prepare metadata
    doc_id = str(uuid.uuid4())
    chunk_metadata = []
    chunk_ids = []
    
    for i, chunk in enumerate(chunks):
        chunk_metadata.append({
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "doc_id": doc_id,
            "upload_time": datetime.now().isoformat(),
            **(metadata or {})
        })
        chunk_ids.append(f"{doc_id}_chunk_{i}")
    
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=chunk_metadata,
        ids=chunk_ids
    )
    
    return doc_id

def search_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
    """Search for relevant document chunks using semantic similarity."""
    if embedding_model is None:
        # Fallback to keyword search only
        return search_keyword_chunks(query, top_k)
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query])
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    
    # Format results
    relevant_chunks = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            relevant_chunks.append({
                "content": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results['distances'] else 0
            })
    
    return relevant_chunks

def search_keyword_chunks(query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
    """Search for relevant document chunks using keyword matching."""
    try:
        # Search in ChromaDB using text query
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        relevant_chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                relevant_chunks.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        return relevant_chunks
    except Exception as e:
        print(f"Warning: Keyword search failed: {e}")
        return []

def hybrid_search(query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
    """Perform hybrid search combining semantic and keyword search."""
    if embedding_model is None:
        # Fallback to keyword search only
        return search_keyword_chunks(query, top_k)
    
    # Semantic search
    semantic_results = search_relevant_chunks(query, top_k)
    
    # Keyword search (simple implementation)
    try:
        keyword_results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Combine and deduplicate results
        all_results = []
        seen_contents = set()
        
        # Add semantic results
        for result in semantic_results:
            if result["content"] not in seen_contents:
                all_results.append(result)
                seen_contents.add(result["content"])
        
        # Add keyword results
        if keyword_results['documents'] and keyword_results['documents'][0]:
            for i, doc in enumerate(keyword_results['documents'][0]):
                if doc not in seen_contents:
                    all_results.append({
                        "content": doc,
                        "metadata": keyword_results['metadatas'][0][i] if keyword_results['metadatas'] else {},
                        "distance": keyword_results['distances'][0][i] if keyword_results['distances'] else 0
                    })
                    seen_contents.add(doc)
        
        # Sort by relevance (lower distance = more relevant)
        all_results.sort(key=lambda x: x.get("distance", 1.0))
        
        return all_results[:top_k]
    except Exception as e:
        print(f"Warning: Hybrid search failed, using semantic only: {e}")
        return semantic_results

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

    # Pattern matching for medical queries
    medical_patterns = [
        r"\b(i have|i feel|i'm feeling|i am feeling)\b",
        r"\b(should i|can i|do i need|when should i)\b",
        r"\b(what are the symptoms|how do i treat|how to treat|what is)\b",
        r"\b(side effects? of|effects? of|reaction to)\b",
        r"\b(my|the) (head|stomach|back|chest|throat|ear|eye|nose|skin)\b",
        r"\b(pain|ache|hurt|sore|burning|throbbing|sharp|dull)\b",
        r"\b(fever|temperature|hot|cold|chills|sweating)\b",
        r"\b(nausea|vomiting|dizzy|weak|tired|fatigue)\b",
        r"\b(cough|sneeze|runny|congestion|breathing)\b",
        r"\b(rash|itch|burn|swelling|lump|bump)\b",
        r"\b(running nose|runny nose|stuffy nose|nasal congestion)\b"
    ]
    
    for pattern in medical_patterns:
        if re.search(pattern, msg):
            return "medical"

    return "irrelevant"

# --- API Endpoints ---
@app.post('/upload')
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for RAG."""
    try:
        # Extract text
        text = extract_text_from_file(file)
        
        # Add to vector database
        doc_id = add_document_to_vector_db(text, file.filename)
        
        # Generate summary
        summary_prompt = f"Summarize the following medical document in 3-5 key points:\n\n{text[:1000]}..."
        chat_history = [
            {"role": "system", "content": "You are a medical assistant. Provide concise, accurate summaries."},
            {"role": "user", "content": summary_prompt}
        ]
        
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            temperature=0.3,
            max_tokens=300
        )
        summary = completion.choices[0].message.content.strip()
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "summary": summary,
            "message": f"Document '{file.filename}' uploaded and processed successfully!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint that uses classify_message for greetings and medical queries, and RAG for document queries."""
    try:
        data = await request.json()
        user_message = data.get("message", "").strip()
        model = data.get("model") or "llama3:latest"
        doc_id = data.get("doc_id")

        if not user_message:
            return JSONResponse({"response": "⚠️ Please enter a message."}, status_code=400)

        # If doc_id is provided and not empty, use RAG to answer from the document
        print(f"[DEBUG] Received doc_id: '{doc_id}' (type: {type(doc_id)})")
        if doc_id and doc_id.strip():
            # Retrieve all chunks for this document
            results = collection.get(where={"doc_id": doc_id})
            document_chunks = results.get('documents', [])
            
            if document_chunks:
                all_chunks = [
                    {"content": doc, "metadata": results['metadatas'][i]}
                    for i, doc in enumerate(document_chunks)
                ]
                
                # Use semantic similarity to find the most relevant chunks from this document only
                from sentence_transformers.util import cos_sim
                query_embeddings = embedding_model.encode([user_message])
                doc_embeddings = embedding_model.encode([chunk['content'] for chunk in all_chunks])
                
                # Compute similarity and get top chunks
                import numpy as np
                similarities = cos_sim(query_embeddings, doc_embeddings).cpu().numpy()[0]
                top_indices = similarities.argsort()[-5:][::-1]  # Top 5 chunks
                context_chunks = [all_chunks[i] for i in top_indices if all_chunks[i]['content'].strip()]
                context_text = "\n".join([chunk['content'] for chunk in context_chunks])
                
                print(f"[DEBUG] doc_id: {doc_id}")
                print(f"[DEBUG] User question: {user_message}")
                print(f"[DEBUG] Context from document:\n{context_text}\n---")
                
                if not context_text.strip():
                    return {"response": "I cannot find relevant information in your uploaded document to answer this question. Please ask a question that relates to the content of your document."}
                
                # Enhanced system prompt for document-specific answers
                system_prompt = (
                    "You are a medical assistant that answers questions based ONLY on the uploaded document content. "
                    "IMPORTANT RULES:\n"
                    "1. Answer questions using ONLY information from the provided document\n"
                    "2. If the document doesn't contain relevant information, say 'This information is not available in your uploaded document'\n"
                    "3. Do NOT provide general medical advice - stick to what's in the document\n"
                    "4. Be specific and reference exact details from the document\n"
                    "5. If asked about symptoms, treatments, medications, etc., only mention what's explicitly stated in the document\n"
                    "6. Always start your response with 'Based on your document:'\n\n"
                    "Document Content:\n" + context_text
                )
                
                chat_history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=chat_history,
                    temperature=0.3,  # Lower temperature for more focused answers
                    max_completion_tokens=1024,
                    top_p=0.9,
                    stream=False,
                    stop=None
                )
                
                response_text = completion.choices[0].message.content.strip()
                
                # Validate that the response is document-specific
                if not response_text or any(phrase in response_text.lower() for phrase in [
                    'general information', 'general overview', 'not sure', 'i don\'t know',
                    'general medical advice', 'common symptoms', 'typical treatment'
                ]):
                    response_text = "This information is not available in your uploaded document. Please ask a question that relates to the specific content of your document."
                
                return {"response": response_text}
            else:
                return {"response": "No document content found. Please upload a document first and then ask your question."}

        # Otherwise, use the normal logic
        # Classify the message
        msg_type = classify_message(user_message)
        print(f"[DEBUG] Message: '{user_message}' classified as: {msg_type}")
        print(f"[DEBUG] No valid doc_id provided, using normal chat logic")

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
            # First, try to find relevant information in uploaded documents using RAG
            try:
                # Search for relevant chunks in all documents
                relevant_chunks = hybrid_search(user_message, top_k=3)
                
                if relevant_chunks and embedding_model:
                    # Use RAG with relevant document information
                    context_text = "\n".join([chunk['content'] for chunk in relevant_chunks])
                    system_prompt = (
                        "You are a medical assistant with access to uploaded medical documents. "
                        "When a user describes a symptom or problem, first check if the uploaded documents contain relevant information. "
                        "If relevant information is found in the documents, use it to provide a more accurate response. "
                        "Always provide: brief explanation, common causes, home remedies, and when to seek medical attention. "
                        "If no relevant information is found in documents, provide general medical advice. "
                        "Be clear, medically accurate, and concise.\n\n"
                        f"[Relevant Document Information]\n{context_text}\n\n"
                        "[User Query]\n{user_message}"
                    )
                    chat_history = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                else:
                    # Fallback to general medical knowledge
                    system_prompt = (
                        "You are a medical assistant. When a user describes a symptom or problem, "
                        "always provide a brief explanation, common causes, home remedies, and when to seek medical attention. "
                        "Do not ask for more details unless absolutely necessary."
                    )
                    chat_history = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=chat_history,
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None
                )
                response_text = completion.choices[0].message.content.strip()
                if not response_text:
                    response_text = "Sorry, I could not generate a response. Please try again."
                return {"response": response_text}
            except Exception as e:
                return {"response": f"❌ Error from Groq API: {str(e)}"}
        elif msg_type == "document_query":
            # Optionally, you could add custom logic for document queries here
            try:
                response_text = ollama_client.generate(user_message, model=model)
                return {"response": response_text}
            except Exception as e:
                return {"response": f"❌ Error: {str(e)}"}
        else:
            return {"response": "🤖 I'm here to help with medical, nutrition, and wellness questions. Please ask me anything related to your health!"}

    except Exception as e:
        print(f"[ERROR] /chat endpoint failed: {e}")
        return {"response": "Sorry, there was an error processing your request. Please try again later."}

@app.get("/")
def get_homepage():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    try:
        # Get all documents from ChromaDB
        results = collection.get()
        
        # Group by document ID
        documents = {}
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas']):
                doc_id = metadata.get('doc_id')
                if doc_id not in documents:
                    documents[doc_id] = {
                        'filename': metadata.get('filename', 'Unknown'),
                        'upload_time': metadata.get('upload_time', ''),
                        'chunks': 0
                    }
                documents[doc_id]['chunks'] += 1
        
        return {"documents": list(documents.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document."""
    try:
        # Get all chunks for this document
        results = collection.get(where={"doc_id": doc_id})
        
        if results['ids']:
            # Delete all chunks for this document
            collection.delete(ids=results['ids'])
            return {"message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
