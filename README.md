# NutriMed Bot RAG 🤖

A comprehensive medical chatbot enhanced with **Retrieval-Augmented Generation (RAG)** for intelligent document-based responses.

## ✨ Features

### 🤖 Core Chatbot
- **Medical Q&A**: Ask about symptoms, treatments, medications, nutrition, and exercises
- **Structured Responses**: Organized information with clear sections
- **Smart Classification**: Automatically detects medical vs non-medical queries

### 📄 RAG (Retrieval-Augmented Generation)
- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Vector Embeddings**: Uses sentence transformers for semantic understanding
- **Hybrid Search**: Combines semantic and keyword search for better results
- **Intelligent Chunking**: Splits documents into optimal chunks for retrieval
- **Context-Aware Responses**: Uses uploaded documents to enhance answers

### 🎯 Key Capabilities
- **Multi-format Support**: PDF, DOCX, TXT document processing
- **Persistent Storage**: ChromaDB for efficient vector storage
- **Real-time Processing**: Instant document analysis and embedding
- **Smart Retrieval**: Finds most relevant document chunks for queries
- **Enhanced Accuracy**: Combines general medical knowledge with document context

## 🚀 Quick Start

### 1. Set Environment Variable
```bash
# Windows
set GROQ_API_KEY=your_groq_api_key_here

# Linux/Mac
export GROQ_API_KEY=your_groq_api_key_here
```

### 2. Start the Server
```bash
# Option 1: Use the startup script (recommended)
python start_server.py

# Option 2: Manual installation
pip install -r requirements.txt
python main.py
```

### 3. Open the Interface
Open `index.html` in your browser or serve it with a local server.

## 📋 Usage Guide

### Basic Medical Queries
Simply type your medical questions:
- "I have a headache"
- "What are the symptoms of diabetes?"
- "How to treat joint pain?"

### Document-Based Queries
1. **Upload a Document**: Click the 📄 icon to upload medical documents
2. **Ask Questions**: Ask about the uploaded content:
   - "Summarize this document"
   - "What does it say about treatment options?"
   - "Explain the key findings"

### RAG Features
- **Automatic Context**: The bot automatically uses relevant document chunks
- **Hybrid Responses**: Combines document information with general medical knowledge
- **Smart Filtering**: Only uses relevant document sections for each query

## 🛠️ Technical Architecture

### Backend Components
- **FastAPI**: Modern web framework for API endpoints
- **Groq**: High-performance LLM for text generation
- **Sentence Transformers**: Semantic embeddings for document understanding
- **ChromaDB**: Vector database for efficient similarity search
- **PyMuPDF**: PDF text extraction
- **python-docx**: DOCX document processing

### RAG Pipeline
1. **Document Processing**: Extract and clean text from uploaded files
2. **Chunking**: Split documents into overlapping chunks (512 words, 50 overlap)
3. **Embedding**: Generate vector embeddings using all-MiniLM-L6-v2
4. **Storage**: Store chunks and embeddings in ChromaDB
5. **Retrieval**: Hybrid search (semantic + keyword) for relevant chunks
6. **Generation**: Use retrieved context to enhance LLM responses

### API Endpoints
- `POST /chat`: Main chat endpoint with RAG capabilities
- `POST /upload`: Document upload and processing
- `GET /documents`: List uploaded documents
- `DELETE /documents/{doc_id}`: Delete specific documents

## 📁 Project Structure
```
NutriMed Bot/
├── main.py              # RAG-enhanced FastAPI server
├── api.py               # Original API (backup)
├── start_server.py      # Startup script
├── requirements.txt     # Python dependencies
├── index.html          # Frontend interface
├── script.js           # Frontend JavaScript
├── style.css           # Frontend styling
├── chroma_db/          # Vector database storage
└── README.md           # This file
```

## 🔧 Configuration

### RAG Settings (in main.py)
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model
CHUNK_SIZE = 512                       # Words per chunk
CHUNK_OVERLAP = 50                     # Overlap between chunks
TOP_K_RESULTS = 3                      # Number of relevant chunks to retrieve
```

### Performance Optimization
- **Lightweight Embeddings**: Uses all-MiniLM-L6-v2 for fast processing
- **Efficient Chunking**: Optimized chunk size for medical documents
- **Persistent Storage**: ChromaDB maintains embeddings between sessions

## 🎯 Use Cases

### Medical Professionals
- Upload patient reports and ask for analysis
- Get quick summaries of medical documents
- Cross-reference multiple documents

### Patients
- Understand medical reports and diagnoses
- Get explanations of medical terminology
- Learn about treatment options

### Researchers
- Process large volumes of medical literature
- Extract key information from research papers
- Generate summaries of medical studies

## 🔒 Privacy & Security
- **Local Processing**: Documents are processed locally
- **No External Storage**: Vector database stored locally
- **Secure API**: Uses environment variables for API keys
- **Data Control**: Users can delete uploaded documents

## 🚨 Important Notes
- **Not Medical Advice**: This is a research tool, not a substitute for professional medical advice
- **API Key Required**: You need a Groq API key for the LLM functionality
- **Document Limits**: Large documents may take time to process
- **Browser Compatibility**: Works best with modern browsers

## 🐛 Troubleshooting

### Common Issues
1. **API Key Error**: Ensure GROQ_API_KEY is set correctly
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Port Already in Use**: Change port in main.py or kill existing process
4. **Document Upload Fails**: Check file format (PDF, DOCX, TXT only)

### Performance Tips
- Use smaller documents for faster processing
- Restart server if memory usage is high
- Clear browser cache if interface issues occur

## 🤝 Contributing
Feel free to contribute improvements:
- Better document processing
- Enhanced RAG algorithms
- UI/UX improvements
- Additional medical knowledge sources

## 📄 License
This project is for educational and research purposes. Please ensure compliance with medical data regulations in your jurisdiction. 