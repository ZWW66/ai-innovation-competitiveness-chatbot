# import sys
# import os

# # Add the project root directory to sys.path
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# sys.path.append(PROJECT_ROOT)


# import streamlit as st
# from crew.main import kickoff_query

# st.set_page_config(page_title="AI Innovation & Competitiveness Chatbot", page_icon="🤖", layout="wide")
# st.title("🤖 AI Innovation & Competitiveness Chatbot")

# # Sidebar controls
# with st.sidebar:
#     st.subheader("How to use")
#     st.markdown(
#         "- Ask questions about AI policy, chips/export controls, compute, talent, and market impacts.\n"
#         "- Answers are grounded in your embedded news articles and include citations.\n"
#         "- If you updated the CSV, re-run: `python -m rag.ingest`."
#     )
#     if st.button("🧹 Clear chat"):
#         st.session_state.history = []

# # Chat history state
# if "history" not in st.session_state:
#     st.session_state.history = []  # list of dicts: {"role": "user"|"assistant", "content": str}

# # Chat input (bottom bar style)
# prompt = st.chat_input("Ask about AI innovation & competitiveness...")
# if prompt:
#     # Show immediately
#     st.session_state.history.append({"role": "user", "content": prompt})
#     with st.spinner("Thinking..."):
#         try:
#             answer = kickoff_query(prompt)  # calls CrewAI pipeline
#         except Exception as e:
#             answer = f"Sorry, something went wrong: `{e}`"
#     st.session_state.history.append({"role": "assistant", "content": str(answer)})

# # Render conversation
# for msg in st.session_state.history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Optional quick starters
# st.divider()
# st.caption("Quick examples:")
# cols = st.columns(3)
# examples = [
#     "How do recent AI chip export controls affect global competitiveness?",
#     "Summarize the latest AI datacenter investments and their implications.",
#     "What are current trends in AI regulation that impact innovation?"
# ]
# for i, ex in enumerate(examples):
#     if cols[i % 3].button(ex):
#         st.session_state.history.append({"role": "user", "content": ex})
#         with st.spinner("Thinking..."):
#             try:
#                 answer = kickoff_query(ex)
#             except Exception as e:
#                 answer = f"Sorry, something went wrong: `{e}`"
#         st.session_state.history.append({"role": "assistant", "content": str(answer)})
#         st.rerun()
# frontend/app.py
import sys
import os

# Add the project root directory to sys.path (so `crew.*` imports work when running Streamlit)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
from crew.main import kickoff_query                   # expects signature: kickoff_query(query: str, domain_directive: str)
from crew.tasks import DOMAIN_DIRECTIVES              # dict of domain -> directive text

st.set_page_config(page_title="AI Innovation & Competitiveness Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Innovation & Competitiveness Chatbot")

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("How to use")
    st.markdown(
        "- Ask questions about AI policy, chips/export controls, compute, talent, and market impacts.\n"
        "- Answers are grounded in embedded news articles with citations.\n"
        "- If you updated the CSV, re-run: `python -m rag.ingest`."
    )

    # Domain selector (Option A)
    selected_domain = st.selectbox(
        "Domain focus",
        ["general", "policy", "research", "product", "manufacturing"],
        index=0,
        help="Tailor the answer style and focus to this AI sub-domain."
    )

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.history = []

# ---------- Chat state ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict]: {"role": "user"|"assistant", "content": str}

# ---------- Chat input ----------
prompt = st.chat_input("Ask about AI innovation & competitiveness...")
if prompt:
    # show user message immediately
    st.session_state.history.append({"role": "user", "content": prompt})

    directive = DOMAIN_DIRECTIVES[selected_domain]
    with st.spinner("Thinking..."):
        try:
            answer = kickoff_query(query=prompt, domain_directive=directive)
        except Exception as e:
            answer = f"Sorry, something went wrong: `{e}`"

    st.session_state.history.append({"role": "assistant", "content": str(answer)})

# ---------- Render conversation ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Quick starters ----------
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

        directive = DOMAIN_DIRECTIVES[selected_domain]
        with st.spinner("Thinking..."):
            try:
                answer = kickoff_query(query=ex, domain_directive=directive)
            except Exception as e:
                answer = f"Sorry, something went wrong: `{e}`"

        st.session_state.history.append({"role": "assistant", "content": str(answer)})
        st.rerun()
