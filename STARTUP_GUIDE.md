# 🚀 NutriMed Bot - Startup Guide

## Quick Start

### 1. Set Your API Key
```bash
# Windows
set GROQ_API_KEY=your_groq_api_key_here

# Linux/Mac  
export GROQ_API_KEY=your_groq_api_key_here
```

### 2. Start the Server
```bash
python start_simple.py
```

### 3. Open the Interface
- Open `index.html` in your browser
- Or serve it with a local server

## 🎯 How It Works Now

### **Document-Aware Chatbot**
- **Upload a document** → It becomes your "current session document"
- **Ask any question** → Bot will ALWAYS try to relate it to your uploaded document
- **Continue chatting** → All questions will reference your document until you start a new chat
- **New Chat** → Click "+ New Chat" to reset and upload a different document

### **Example Workflow**
1. **Upload**: Medical report about diabetes
2. **Ask**: "What are the symptoms?" → Bot answers based on your diabetes report
3. **Ask**: "How to treat this?" → Bot references your specific document
4. **Ask**: "What about diet?" → Bot connects diet advice to your diabetes document
5. **New Chat**: Click "+ New Chat" → Upload a different document (e.g., heart disease report)
6. **Ask**: "What are the risks?" → Bot now answers based on the heart disease document

### **Key Features**
- ✅ **Session Persistence**: Document context stays active throughout conversation
- ✅ **Smart Prioritization**: Current document gets priority over other uploaded docs
- ✅ **Contextual Responses**: All answers try to reference your uploaded document
- ✅ **Easy Reset**: "+ New Chat" button clears document context
- ✅ **Document Summary**: Automatic summary when you upload

## 🔧 Troubleshooting

### Server Won't Start?
```bash
# Check if dependencies are installed
pip install groq fastapi pydantic uvicorn python-multipart PyMuPDF python-docx
```

### Upload Fails?
- Make sure server is running on `http://localhost:8000`
- Check file format (PDF, DOCX, TXT only)
- Check file size (not too large)

### Bot Not Responding?
- Check browser console for errors
- Verify GROQ_API_KEY is set correctly
- Restart the server

## 📝 Usage Tips

1. **Upload First**: Always upload your document before asking questions
2. **Ask Naturally**: Ask questions as you normally would - the bot will automatically relate them to your document
3. **Be Specific**: More specific questions get better document-relevant answers
4. **New Document**: Use "+ New Chat" when you want to switch to a different document
5. **Multiple Questions**: You can ask follow-up questions - the bot remembers your document context

## 🎉 You're Ready!

Your chatbot will now:
- Remember your uploaded document throughout the conversation
- Always try to relate answers to your specific document
- Provide contextual, document-aware responses
- Reset cleanly when you start a new chat

Start the server and try uploading a medical document! 