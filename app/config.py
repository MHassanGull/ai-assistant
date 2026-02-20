import os
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

# ===============================
# WEBSITE CONFIG
# ===============================

SITE_URL = os.getenv("SITE_URL", "https://mhassangull.github.io/About-Hassan/")

# ===============================
# PINECONE CONFIG
# ===============================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "website-chatbot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE = os.getenv("NAMESPACE", "portfolio-ai-v1")

# ===============================
# HUGGINGFACE CONFIG
# ===============================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

HF_CHAT_MODEL_CANDIDATES = [
    "HuggingFaceTB/SmolLM3-3B:hf-inference",
    "Qwen/Qwen2.5-1.5B-Instruct:hf-inference",
    "google/gemma-2-2b-it:hf-inference",
]

# ===============================
# EMBEDDING CONFIG
# ===============================

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384

# ===============================
# RAG SETTINGS
# ===============================

TOP_K = 4
CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
UPSERT_BATCH = 60

# ===============================
# SYSTEM PROMPT
# ===============================

SYSTEM_PROMPT = (
    "You are the AI assistant for this portfolio website. "
    "Answer ONLY using the provided website context. "
    "If the answer is not found, say: "
    "'I don’t have that information from the website yet.' "
    "Do NOT include reasoning steps, analysis, or <think> blocks. "
    "Provide only the final answer in a clean and professional format."
)
