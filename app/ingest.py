import re
import hashlib
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from .config import SITE_URL, CHUNK_SIZE, CHUNK_OVERLAP, UPSERT_BATCH
from .embeddings import embed_documents
from .vector_store import upsert_documents, namespace_has_data, clear_namespace


BASE_URL = SITE_URL.rstrip("/")


# ===============================
# Utilities
# ===============================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks


def chunk_id(url: str, chunk: str, idx: int) -> str:
    h = hashlib.sha1((url + "\n" + chunk).encode("utf-8")).hexdigest()[:24]
    return f"{idx}-{h}"


# ===============================
# Link Filtering
# ===============================

def is_internal_link(link: str) -> bool:
    if not link:
        return False

    if link.startswith("#"):
        return False

    if link.startswith("mailto:") or link.startswith("tel:"):
        return False

    full_url = urljoin(SITE_URL, link)
    return full_url.startswith(BASE_URL)


def extract_links(soup):
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"]

        if is_internal_link(href):
            full = urljoin(SITE_URL, href)
            clean = full.split("#")[0].rstrip("/")
            links.add(clean)

    return links


# ===============================
# Fetch Single Page
# ===============================

def fetch_page(url: str, client):
    response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text_content = soup.get_text(" ")

    # Extract links
    links_text = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        if not href or href.startswith("#"):
            continue

        anchor_text = tag.get_text(strip=True)

        if anchor_text:
            links_text.append(f"{anchor_text} → {href}")
        else:
            links_text.append(href)

    combined_text = text_content + "\n\n" + "\n".join(links_text)

    return clean_text(combined_text), soup


# ===============================
# Multi-page Crawler
# ===============================

def crawl_website():
    visited = set()
    to_visit = {BASE_URL}
    all_text = []

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:

        while to_visit:
            url = to_visit.pop()

            if url in visited:
                continue

            print(f"🌐 Crawling: {url}")

            try:
                text, soup = fetch_page(url, client)
                visited.add(url)

                if text:
                    all_text.append(text)

                new_links = extract_links(soup)
                to_visit.update(new_links - visited)

            except Exception as e:
                print(f"⚠️ Failed: {url} → {e}")

    return "\n\n".join(all_text)


# ===============================
# Sync Website
# ===============================

def sync_website(force: bool = False):

    if namespace_has_data() and not force:
        print("ℹ️ Already synced. Skipping.")
        return

    if force:
        clear_namespace()

    print("🚀 Starting multi-page crawl...")
    full_text = crawl_website()

    if len(full_text) < 200:
        raise RuntimeError("Extracted text too small. Check website content.")

    print("🧩 Chunking text...")
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"🔢 Total chunks: {len(chunks)}")

    # Batch upsert
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i:i+UPSERT_BATCH]
        embeddings = embed_documents(batch)

        vectors = []
        for j, (chunk, vector) in enumerate(zip(batch, embeddings), start=i):
            vectors.append(
                (
                    chunk_id(SITE_URL, chunk, j),
                    vector,
                    {
                        "source": SITE_URL,
                        "chunk_index": j,
                        "text": chunk,
                    },
                )
            )

        upsert_documents(vectors)

    print("✅ Website synced successfully.")
