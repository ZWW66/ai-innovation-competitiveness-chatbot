import sys
import os

# Add the project root directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)


import streamlit as st
from crew.main import kickoff_query

st.set_page_config(page_title="AI Innovation & Competitiveness Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Innovation & Competitiveness Chatbot")

# Sidebar controls
with st.sidebar:
    st.subheader("How to use")
    st.markdown(
        "- Ask questions about AI policy, chips/export controls, compute, talent, and market impacts.\n"
        "- Answers are grounded in your embedded news articles and include citations.\n"
        "- If you updated the CSV, re-run: `python -m rag.ingest`."
    )
    if st.button("🧹 Clear chat"):
        st.session_state.history = []

# Chat history state
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role": "user"|"assistant", "content": str}

# Chat input (bottom bar style)
prompt = st.chat_input("Ask about AI innovation & competitiveness...")
if prompt:
    # Show immediately
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            answer = kickoff_query(prompt)  # calls CrewAI pipeline
        except Exception as e:
            answer = f"Sorry, something went wrong: `{e}`"
    st.session_state.history.append({"role": "assistant", "content": str(answer)})

# Render conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Optional quick starters
st.divider()
st.caption("Quick examples:")
cols = st.columns(3)
examples = [
    "How do recent AI chip export controls affect global competitiveness?",
    "Summarize the latest AI datacenter investments and their implications.",
    "What are current trends in AI regulation that impact innovation?"
]
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex):
        st.session_state.history.append({"role": "user", "content": ex})
        with st.spinner("Thinking..."):
            try:
                answer = kickoff_query(ex)
            except Exception as e:
                answer = f"Sorry, something went wrong: `{e}`"
        st.session_state.history.append({"role": "assistant", "content": str(answer)})
        st.experimental_rerun()
