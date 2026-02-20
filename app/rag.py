from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception

from .config import (
    HF_TOKEN,
    HF_ROUTER_BASE_URL,
    HF_CHAT_MODEL_CANDIDATES,
    SYSTEM_PROMPT,
    TOP_K,
)
from .embeddings import embed_query
from .vector_store import query_vectors


# ===============================
# Guards
# ===============================

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in environment variables.")

# ===============================
# Initialize LLM Client
# ===============================

hf_client = OpenAI(
    base_url=HF_ROUTER_BASE_URL,
    api_key=HF_TOKEN,
)


# ===============================
# Retry Logic
# ===============================

def _retryable_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        keyword in msg
        for keyword in ["429", "rate limit", "timeout", "503", "overloaded", "try again"]
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=1, max=18),
    retry=retry_if_exception(_retryable_error),
)
def chat_once(model: str, messages, max_tokens: int = 450, temperature: float = 0.2):
    response = hf_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ===============================
# Pick Working Model
# ===============================

def pick_model():
    last_error = None
    for model in HF_CHAT_MODEL_CANDIDATES:
        try:
            test = chat_once(
                model,
                [{"role": "user", "content": "Reply exactly with OK"}],
                max_tokens=10,
                temperature=0.0,
            )
            if "ok" in test.lower():
                print(f"✅ Using model: {model}")
                return model
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"No working HF model found. Last error: {last_error}")


HF_CHAT_MODEL = pick_model()


# ===============================
# Retrieval
# ===============================

def retrieve_context(question: str, top_k: int = TOP_K) -> str:
    query_vector = embed_query(question)
    matches = query_vectors(query_vector, top_k=top_k)

    contexts = []
    for match in matches:
        metadata = match.get("metadata", {})
        text = metadata.get("text")
        if text:
            contexts.append(text)

    return "\n\n---\n\n".join(contexts)


# ===============================
# Final Answer Function
# ===============================

def answer(question: str, memory=None) -> str:

    # Retrieve website context first (RAG priority)
    context = retrieve_context(question)

    user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject recent conversation memory (if exists)
    if memory:
        messages.extend(memory.get_recent_messages())

    # Add new question
    messages.append({"role": "user", "content": user_prompt})

    response = chat_once(HF_CHAT_MODEL, messages)

    return response
