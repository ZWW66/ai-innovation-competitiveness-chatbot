# crew/llm.py
from dotenv import load_dotenv

load_dotenv()

from crewai import LLM

chatgpt_llm = LLM(
    model="openai/gpt-5.6-luna",
    reasoning_effort="none",
    max_completion_tokens=1500,
)
