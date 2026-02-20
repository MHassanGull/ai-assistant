from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBED_MODEL_ID, EXPECTED_DIM

# ===============================
# Initialize Embedding Model
# ===============================

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True},
)

# Dimension validation
_dim_check = len(_embeddings.embed_query("dimension check"))

if _dim_check != EXPECTED_DIM:
    raise RuntimeError(
        f"Embedding dimension mismatch: expected {EXPECTED_DIM}, got {_dim_check}"
    )

print(f"✅ Embeddings loaded successfully (dim={_dim_check})")


# ===============================
# Public Functions
# ===============================

def embed_query(text: str):
    """
    Embed a single query string.
    """
    return _embeddings.embed_query(text)


def embed_documents(texts: list):
    """
    Embed a list of text documents.
    """
    return _embeddings.embed_documents(texts)
