# crew/llm.py
from dotenv import load_dotenv
load_dotenv()

from crewai import LLM

chatgpt_llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.2,
)
