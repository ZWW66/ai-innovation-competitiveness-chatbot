# AI Innovation & Competitiveness Chatbot

A domain-specific chatbot for questions about AI innovation, industrial competitiveness, policy, compute, talent, and market developments. Answers are grounded in an RSS-derived news corpus and include source links.

## Architecture

- **CrewAI** orchestrates a News Researcher and an AI Competitiveness Expert.
- **LangChain RAG** retrieves relevant news passages.
- **Sentence Transformers** (`all-MiniLM-L6-v2`) creates local embeddings.
- **Chroma** stores the persistent vector index.
- **GPT-5.6 Luna** generates cost-conscious answers.
- **Streamlit** provides the chat interface.

## Project Structure

```text
ai-innovation-competitiveness-chatbot/
├── crew/                 # Agents, tasks, tools, and LLM configuration
├── data/
│   ├── news_articles_combined_*.csv
│   └── vectorstore_news_ai/
├── frontend/app.py       # Streamlit application
├── news_ingestion/       # RSS/Atom collection pipeline
├── rag/                  # Embeddings, ingestion, and retrieval
├── tests/
├── .env.example
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```dotenv
OPENAI_API_KEY=sk-xxxx...
```

The chatbot uses `gpt-5.6-luna` with reasoning disabled and a 1,500-token output cap.

## Run

If the included vectorstore is present, launch the application directly:

```bash
streamlit run frontend/app.py
```

The backend smoke test is optional:

```bash
python -m crew.main
```

## Refresh the News Corpus

Fast RSS/Atom collection is the default:

```bash
python -m news_ingestion.scrape_news
```

Useful alternatives:

```bash
# Small reproducibility check without merging existing data
python -m news_ingestion.scrape_news --max-per-source 5 --no-merge

# Slower run that downloads missing article bodies
python -m news_ingestion.scrape_news --full-text
```

The scraper uses 35 AI-focused sources and provides bounded concurrency, short retries, relevance filtering, URL normalization, deduplication, per-host throttling, and incremental CSV merging.

Rebuild the vectorstore after refreshing the CSV:

```bash
python -m rag.ingest
```

Ingestion recreates the Chroma collection from scratch, preventing duplicate chunks across repeated builds.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- `.env` and local virtual environments are excluded from Git.
- Streamlit file watching is disabled because Transformers exposes optional vision modules that this text-only application does not use.
- The included CSV and vectorstore make the application runnable after cloning; both can be regenerated from the documented pipeline.
