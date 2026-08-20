# rag/retriever.py
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
# rag/retriever.py
from langchain_community.vectorstores import Chroma  # stable across versions

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "vectorstore_news_ai")

class STEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

def get_retriever(k: int = 5, model_name: str = "all-MiniLM-L6-v2"):
    embeddings = STEmbeddings(model_name=model_name)
    vectordb = Chroma(
        collection_name="news-ai",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})
