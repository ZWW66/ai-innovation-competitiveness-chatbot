import os

from langchain_chroma import Chroma

from rag.embeddings import SentenceTransformerEmbeddings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore_news_ai")


def get_retriever(k: int = 5, model_name: str = "all-MiniLM-L6-v2"):
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    vectordb = Chroma(
        collection_name="news-ai",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})
