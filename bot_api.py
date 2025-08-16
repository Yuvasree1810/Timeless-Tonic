import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set. Please set it before running this script.")

client = Groq(api_key=api_key)

greeting_keywords = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}

def is_greeting(text):
    return any(word in text.lower() for word in greeting_keywords)

general_keywords = {"yes", "ok", "okay", "thanks", "thank you", "sure", "alright", "fine", "cool", "great"}

def is_general(text):
    return any(word in text.lower() for word in general_keywords)

class ChatRequest(BaseModel):
    symptom: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_symptom = request.symptom.strip()
    if is_greeting(user_symptom):
        greeting_response = (
            "Hi, how may I help you?"
        )
        suggestions = ["What are common symptoms of headache?", "How can I improve my nutrition?", "Suggest some exercises for wellness."]
        return JSONResponse(content={"responses": [greeting_response], "suggestions": suggestions})
    if is_general(user_symptom):
        general_response = (
            "I'm here to help! Let me know if you have a health question or need advice."
        )
        suggestions = ["What are the impacts of stress?", "List medications for cold.", "Show me wellness tips."]
        return JSONResponse(content={"responses": [general_response], "suggestions": suggestions})
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are NutriMed Bot, a friendly, helpful health assistant. "
                    "When the user asks for symptoms, recipes, remedies, or instructions, respond with a single message, but format your answer so that each step, ingredient, or point is on its own line. "
                    "Do NOT combine multiple items into a single paragraph or run them together. "
                    "Use clear, bold section headings with emojis where appropriate, and then list each item or step on a new line, like this:\n\n"
                    "## ✨ Turmeric Ginger Tea\n"
                    "1. 1 teaspoon of turmeric powder\n"
                    "2. 1 teaspoon of freshly grated ginger\n"
                    "3. 1 cup of boiling water\n"
                    "4. Honey to taste (optional)\n"
                    "5. Combine turmeric and ginger in a cup, add boiling water, and let it steep for 5-7 minutes.\n"
                    "6. Strain and drink 2-3 times a day.\n\n"
                    "Do NOT use paragraph-style blocks. Each point or step must be on its own line for easy reading and clarity."
                )
            },
            {
                "role": "user",
                "content": user_symptom
            }
        ],
        temperature=0.8,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    import json
    response_text = completion.choices[0].message.content
    try:
        response_json = json.loads(response_text)
        return JSONResponse(content=response_json)
    except Exception:
        # fallback: split by numbered points (1., 2., 3., etc.)
        numbered_split = re.split(r"^\d+\. ", response_text, flags=re.MULTILINE)
        messages = []
        # If the first message is a heading, add it
        if numbered_split[0].strip():
            messages.append(numbered_split[0].strip())
        # Add each numbered item as a separate message
        for i in range(1, len(numbered_split)):
            content = numbered_split[i].strip()
            if content:
                messages.append(f"{i}. {content}")
        return JSONResponse(content={"responses": messages, "suggestions": []}) 