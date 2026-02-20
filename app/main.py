from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from .rag import answer
from .memory import ConversationMemory

# ===============================
# App Initialization
# ===============================

app = FastAPI(
    title="Portfolio AI Assistant",
    version="1.0.0"
)

# ===============================
# CORS (Allow GitHub Pages)
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For now allow all (we restrict later if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Simple Rate Limiter
# ===============================

RATE_LIMIT = 10  # max requests
TIME_WINDOW = 60  # per 60 seconds
request_log = {}

def check_rate_limit(ip: str):
    now = time.time()

    if ip not in request_log:
        request_log[ip] = []

    # Remove expired timestamps
    request_log[ip] = [
        timestamp for timestamp in request_log[ip]
        if now - timestamp < TIME_WINDOW
    ]

    if len(request_log[ip]) >= RATE_LIMIT:
        return False

    request_log[ip].append(now)
    return True


# ===============================
# Request Model
# ===============================

class ChatRequest(BaseModel):
    question: str


# ===============================
# Endpoints
# ===============================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request, data: ChatRequest):

    client_ip = request.client.host

    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

    try:
        # Session-based memory (per request for now)
        memory = ConversationMemory(max_turns=4)

        memory.add_user_message(data.question)

        response = answer(data.question, memory=memory)

        memory.add_assistant_message(response)

        return {
            "success": True,
            "response": response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
