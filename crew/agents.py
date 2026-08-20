from crewai import Agent

from crew.llm import chatgpt_llm
from crew.tools import (
    extract_keywords,
    retrieve_citations,
    retrieve_context,
    summarize_text,
)

news_researcher = Agent(
    role="News Researcher",
    goal="Retrieve and curate the most relevant snippets from the news vector store.",
    backstory="You scan recent news for AI innovation & competitiveness topics.",
    llm=chatgpt_llm,
    tools=[retrieve_context, retrieve_citations, summarize_text, extract_keywords],
    verbose=True,
)

domain_expert = Agent(
    role="AI Competitiveness Expert",
    goal=("Answer strictly with retrieved context and clear citations. "
          "Emphasize policy, market, talent, compute, and ecosystem impacts."),
    backstory="An analyst specializing in AI innovation and competitiveness.",
    llm=chatgpt_llm,
    verbose=True,
)
