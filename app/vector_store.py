import time
from pinecone import Pinecone, ServerlessSpec
from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    NAMESPACE,
    EXPECTED_DIM,
)

# ===============================
# Guards
# ===============================

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment variables.")

# ===============================
# Initialize Pinecone
# ===============================

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = {i["name"] for i in pc.list_indexes()}

if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"🚀 Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EXPECTED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        deletion_protection="disabled",
    )

    # Wait until index is ready
    while True:
        desc = pc.describe_index(PINECONE_INDEX_NAME)
        if desc.status.get("ready"):
            break
        time.sleep(2)

# Validate dimension
desc = pc.describe_index(PINECONE_INDEX_NAME)

if desc.dimension != EXPECTED_DIM:
    raise RuntimeError(
        f"Pinecone dimension mismatch: index={desc.dimension}, expected={EXPECTED_DIM}"
    )

index = pc.Index(PINECONE_INDEX_NAME)

print(f"✅ Pinecone ready (dim={desc.dimension})")

# ===============================
# Public Functions
# ===============================

def upsert_documents(vectors: list):
    """
    vectors format:
    [
        (id, embedding_vector, metadata_dict),
        ...
    ]
    """
    index.upsert(vectors=vectors, namespace=NAMESPACE)


def query_vectors(vector: list, top_k: int = 4):
    """
    Query Pinecone for similar vectors.
    """
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    return result.get("matches", [])


def namespace_has_data() -> bool:
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    return (namespaces.get(NAMESPACE, {}).get("vector_count", 0) or 0) > 0


def clear_namespace():
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})

    if NAMESPACE in namespaces:
        index.delete(delete_all=True, namespace=NAMESPACE)
        print("🧹 Namespace cleared.")
    else:
        print("ℹ️ Namespace does not exist. Nothing to clear.")