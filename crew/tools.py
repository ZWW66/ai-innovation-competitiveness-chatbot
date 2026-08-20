# crew/tools.py
from __future__ import annotations
from typing import List
import re

from crewai.tools import tool                 # ← CrewAI's tool decorator
from langchain_core.documents import Document
from rag.retriever import get_retriever

_retriever = get_retriever(k=5)

def _retrieve_docs(query: str) -> List[Document]:
    """Return top-k retrieved documents using modern LCEL API."""
    return _retriever.invoke(query)

@tool("retrieve_context")
def retrieve_context(query: str) -> str:
    """Given a user query, return one concatenated string of the top-k retrieved news chunks."""
    docs = _retrieve_docs(query)
    return "\n\n".join((d.page_content or "").strip() for d in docs if d.page_content)

@tool("retrieve_citations")
def retrieve_citations(query: str) -> str:
    """Given a user query, return bulleted cited snippets with [title](link)."""
    docs = _retrieve_docs(query)
    lines = []
    for d in docs:
        title = d.metadata.get("title", "") or d.metadata.get("source", "")
        link  = d.metadata.get("link", "")
        snippet = (d.page_content or "").strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        lines.append(f"- {snippet}\n  [Source: {title}]({link})")
    return "\n".join(lines)

def _summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
    return " ".join(sentences[:max_sentences])

@tool("summarize_text")
def summarize_text(text: str) -> str:
    """Given text, return a short extractive summary (first key sentences)."""
    return _summarize_text(text)

def _extract_keywords(text: str, top_k: int = 12) -> str:
    stop = set("""
    the a an and or of for to from in on at as by with without into about over under
    this that those these it its is are was were be being been can could should would
    have has had not no yes if then else when while after before during more most less
    than between within across new latest updated update etc
    """.split())
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    freq = {}
    for t in tokens:
        if t in stop: continue
        freq[t] = freq.get(t, 0) + 1
    return ", ".join([w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]])

@tool("extract_keywords")
def extract_keywords(text: str) -> str:
    """Given text, return comma-separated keywords."""
    return _extract_keywords(text)
