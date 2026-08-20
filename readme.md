# AI Innovation & Competitiveness Chatbot
### CrewAI • LangChain RAG • Sentence Transformers • Streamlit UI

This project implements a **domain-specific expert chatbot** that answers questions about **AI innovation, competitiveness, and industry developments**, using **real news articles** pulled from RSS feeds.

It integrates:

- ✅ CrewAI multi-agent orchestration
- ✅ LangChain RAG pipeline
- ✅ Sentence Transformers embeddings
- ✅ Chroma vector database
- ✅ Streamlit web UI
- ✅ Advanced prompting + context injection
- ✅ Custom RSS news ingestion + text extraction

---

## 📁 Project Structure

```
Group_Project/
├── data/
│   ├── news_articles_*.csv
│   └── vectorstore_news_ai/
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   └── __init__.py
├── crew/
│   ├── agents.py
│   ├── tools.py
│   ├── tasks.py
│   ├── llm.py
│   ├── main.py
│   └── __init__.py
├── frontend/
│   ├── app.py
│   └── __init__.py
├── news_ingestion/
│   ├── scrape_news.py
│   └── ...
├── .env
├── requirements.txt
└── README.md
```

---

## ✅ Installation Instructions

### 1️⃣ Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies
```bash
pip install -U \
  crewai "crewai[openai]" \
  langchain langchain-core langchain-community \
  sentence-transformers \
  chromadb \
  python-dotenv \
  streamlit \
  feedparser \
  beautifulsoup4 lxml \
  trafilatura \
  newspaper3k \
  pandas \
  tqdm \
  requests
```

---

## ✅ 3️⃣ Add Your OpenAI API Key
Create a `.env` file:

```
OPENAI_API_KEY=sk-xxxx...
```

---

# ✅ How to Run the Project

## 🚀 Step 1 — Build the Vectorstore
```bash
python -m rag.ingest
```

## 🚀 Step 2 — Test the CrewAI Backend
```bash
python crew/main.py
```

## 🚀 Step 3 — Launch Streamlit
```bash
streamlit run frontend/app.py
```

---

# 🧠 Advanced Prompting & Context Injection

- Multi-step task instructions
- Forced tool usage
- Strict citation rules
- Automatic injection of RAG context into agent prompts

---

# ✅ Notes for Instructor / TA
- `.env` is excluded
- Vectorstore is reproducible
- Streamlit UI directly calls CrewAI
- Implements multi-agent + RAG + advanced prompting
