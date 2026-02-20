from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import uuid

from .rag import answer
from .memory import ConversationMemory


# ===============================
# App Initialization
# ===============================

app = FastAPI(
    title="Portfolio AI Assistant",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "AI Assistant is running 🚀"
    }


# ===============================
# CORS (Allow GitHub Pages)
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# Rate Limiter
# ===============================

RATE_LIMIT = 10  # max requests per window
TIME_WINDOW = 60  # seconds
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
# Session Memory Storage
# ===============================

# Stores session_id -> ConversationMemory
sessions = {}


# ===============================
# Request Model
# ===============================

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


# ===============================
# Health Endpoint
# ===============================

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ===============================
# Chat Endpoint
# ===============================

@app.post("/chat")
async def chat(request: Request, data: ChatRequest):

    client_ip = request.client.host

    # Rate limit check
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )

    try:
        # ---------------------------
        # Session Handling
        # ---------------------------

        # If frontend did not send session_id → generate one
        if not data.session_id:
            data.session_id = str(uuid.uuid4())

        # If this session does not exist → create memory
        if data.session_id not in sessions:
            sessions[data.session_id] = ConversationMemory(max_turns=4)

        memory = sessions[data.session_id]

        # ---------------------------
        # Add user message
        # ---------------------------
        memory.add_user_message(data.question)

        # ---------------------------
        # Generate answer
        # ---------------------------
        response = answer(data.question, memory=memory)

        # ---------------------------
        # Add assistant message
        # ---------------------------
        memory.add_assistant_message(response)

        # ---------------------------
        # Return response
        # ---------------------------
        return {
            "success": True,
            "session_id": data.session_id,
            "response": response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
