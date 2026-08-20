from __future__ import annotations

import re
import threading

from crewai.tools import tool
from langchain_core.documents import Document

from rag.retriever import get_retriever

STOP_WORDS = {
    "a", "about", "across", "after", "an", "and", "are", "as", "at",
    "be", "been", "before", "being", "between", "by", "can", "could",
    "during", "else", "etc", "for", "from", "had", "has", "have", "if",
    "in", "into", "is", "it", "its", "latest", "less", "more", "most",
    "new", "no", "not", "of", "on", "or", "over", "should", "than",
    "that", "the", "then", "these", "this", "those", "to", "under",
    "update", "updated", "was", "were", "when", "while", "with",
    "within", "without", "would", "yes",
}

_retriever = get_retriever(k=5)
_retriever_lock = threading.Lock()


def _retrieve_docs(query: str) -> list[Document]:
    """Return top-k documents while serializing the shared native retriever."""
    with _retriever_lock:
        return _retriever.invoke(query)


@tool("retrieve_context")
def retrieve_context(query: str) -> str:
    """Return concatenated text from the top matching news chunks."""
    docs = _retrieve_docs(query)
    return "\n\n".join((doc.page_content or "").strip() for doc in docs if doc.page_content)


@tool("retrieve_citations")
def retrieve_citations(query: str) -> str:
    """Return the top news snippets with source links."""
    lines = []
    for doc in _retrieve_docs(query):
        title = doc.metadata.get("title", "") or doc.metadata.get("source", "")
        link = doc.metadata.get("link", "")
        snippet = (doc.page_content or "").strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        lines.append(f"- {snippet}\n  [Source: {title}]({link})")
    return "\n".join(lines)


def _summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]
    return " ".join(sentences[:max_sentences])


@tool("summarize_text")
def summarize_text(text: str) -> str:
    """Return a short extractive summary."""
    return _summarize_text(text)


def _extract_keywords(text: str, top_k: int = 12) -> str:
    frequencies: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower()):
        if token not in STOP_WORDS:
            frequencies[token] = frequencies.get(token, 0) + 1
    ranked = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(word for word, _ in ranked[:top_k])


@tool("extract_keywords")
def extract_keywords(text: str) -> str:
    """Return comma-separated keywords."""
    return _extract_keywords(text)
