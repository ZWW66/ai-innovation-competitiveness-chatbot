import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st

from crew.main import kickoff_query
from crew.tasks import DOMAIN_DIRECTIVES

st.set_page_config(
    page_title="AI Innovation & Competitiveness Chatbot",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 AI Innovation & Competitiveness Chatbot")

with st.sidebar:
    st.subheader("How to use")
    st.markdown(
        "- Ask questions about AI policy, chips/export controls, compute, talent, and market impacts.\n"
        "- Answers are grounded in embedded news articles with citations.\n"
        "- If you updated the CSV, re-run: `python -m rag.ingest`."
    )
    selected_domain = st.selectbox(
        "Domain focus",
        list(DOMAIN_DIRECTIVES),
        help="Tailor the answer style and focus to this AI sub-domain.",
    )
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []


def answer_query(query: str) -> None:
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        try:
            answer = kickoff_query(
                query=query,
                domain_directive=DOMAIN_DIRECTIVES[selected_domain],
            )
        except Exception as exc:  # noqa: BLE001 - surface backend failures in the UI
            answer = f"Sorry, something went wrong: `{exc}`"
    st.session_state.history.append({"role": "assistant", "content": str(answer)})


prompt = st.chat_input("Ask about AI innovation & competitiveness...")
if prompt:
    answer_query(prompt)

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.divider()
st.caption("Quick examples:")
examples = [
    "How do recent AI chip export controls affect global competitiveness?",
    "Summarize the latest AI datacenter investments and their implications.",
    "What are current trends in AI regulation that impact innovation?",
]
for column, example in zip(st.columns(len(examples)), examples):
    if column.button(example):
        answer_query(example)
        st.rerun()
