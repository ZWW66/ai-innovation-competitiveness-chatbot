# rag/ingest.py
# pip install pandas sentence-transformers chromadb langchain-community tqdm

import os
import re
from glob import glob
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

# ---- Paths ----
PROJECT_ROOT = "."
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Chroma persistent directory (saved inside data/)
CHROMA_DIR = os.path.join(DATA_DIR, "vectorstore_news_ai")

# ---- Utils ----
def _latest_csv(data_dir: str) -> str:
    """Pick the newest news_articles_combined_*.csv from data/."""
    pattern = os.path.join(data_dir, "news_articles_combined_*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs found at {pattern}")
    return files[-1]  # newest by lexicographic timestamp

def _coalesce_text(row) -> str:
    """Prefer full article 'text'; otherwise use title + description. Handles NaN/empty."""
    text = str(row.get("text", "") or "").strip()
    if len(text) >= 50:
        return text
    title = str(row.get("title", "") or "").strip()
    desc  = str(row.get("description", "") or "").strip()
    combo = f"{title}. {desc}".strip().strip(". ")
    return combo

def _chunk(text: str, max_words=350, overlap=50) -> list[str]:
    """Word-based chunking with overlap; returns [] for empty input."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

# ---- Embedding wrapper for SentenceTransformers ----
class STEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalize=True lets inner product behave like cosine in Chroma
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

def build_vectorstore(csv_path: str | None = None,
                      chroma_dir: str = CHROMA_DIR,
                      model_name: str = "all-MiniLM-L6-v2",
                      min_len: int = 20,
                      max_words: int = 350,
                      overlap: int = 50):
    """
    Load CSV from /data, clean + chunk, embed with SentenceTransformers, and
    write a persistent Chroma collection.
    """
    if csv_path is None:
        csv_path = _latest_csv(DATA_DIR)

    print(f"📄 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Build doc_text with fallbacks; handle empty strings and NaN safely
    df["doc_text"] = df.apply(_coalesce_text, axis=1)
    # Drop rows where doc_text is too short/empty
    df = df[df["doc_text"].str.len() >= min_len].reset_index(drop=True)
    print(f"✅ Rows after filtering: {len(df)}")

    # Chunk each article
    docs: list[Document] = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = _chunk(r["doc_text"], max_words=max_words, overlap=overlap)
        for ch in chunks:
            if len(ch) < min_len:
                continue
            docs.append(
                Document(
                    page_content=ch,
                    metadata={
                        "source": str(r.get("source", "") or ""),
                        "title":  str(r.get("title", "") or ""),
                        "published": str(r.get("published", "") or ""),
                        "authors": str(r.get("authors", "") or ""),   # author may be empty; that's fine
                        "domain": str(r.get("domain", "") or ""),
                        "link":   str(r.get("link", "") or ""),
                    }
                )
            )
    print(f"🧩 Total chunks: {len(docs)}")

    # Build / overwrite persistent Chroma collection
    os.makedirs(chroma_dir, exist_ok=True)
    embeddings = STEmbeddings(model_name=model_name)
    # Using from_documents will rebuild the collection each run (good for assignment repeatability)
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name="news-ai",
        persist_directory=chroma_dir,
    )
    vectordb.persist()
    print(f"💾 Chroma vector store saved to: {chroma_dir}")

if __name__ == "__main__":
    build_vectorstore()  # uses latest CSV in /data by default
    print("success")
