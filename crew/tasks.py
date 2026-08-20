from crewai import Task

from crew.agents import domain_expert, news_researcher

SYSTEM_RULES = (
    "You MUST answer strictly using the provided context snippets and cite sources as [title](link). "
    "If a claim is not in the context, say you don't know. Be concise, neutral, and structured. "
    "Write in clear paragraphs: explanation + evidence + implications. "
    "Always END your answer with a single bold sentence that begins with 'Impact on AI:' "
    "summarizing the overall effect in ≤25 words. "
    "If relevance is insufficient, write: **Impact on AI: Not enough evidence in retrieved context.**"
)

DOMAIN_DIRECTIVES = {
    "general": "Focus on models, data, training/inference, evaluation, applications, and risks.",
    "policy": "Emphasize regulation, export controls, standards, safety governance, privacy, and geopolitics.",
    "research": "Emphasize methods, datasets, compute scaling, evaluation, ablations, and limitations.",
    "product": "Emphasize user needs, reliability, latency, cost, privacy, deployment, and ROI.",
    "manufacturing": "Emphasize robotics, vision, QA, predictive maintenance, MES/OEE, throughput, yield, and safety.",
}

task_gather = Task(
    description=(
        "Given the user's query (in {query}):\n"
        "1) Use tool `retrieve_context` to get the raw concatenated text of retrieved chunks.\n"
        "2) Use tool `retrieve_citations` to get a bulleted list of cited snippets.\n"
        "3) Use tool `summarize_text` on the raw text to produce a mini-summary.\n"
        "4) Use tool `extract_keywords` on the raw text to produce 8–12 keywords.\n"
        "Return a JSON-like block with keys: 'context_citations', 'summary', 'keywords'."
    ),
    expected_output="A cited context pack + short summary + keywords.",
    agent=news_researcher,
)

task_answer = Task(
    description=(
        f"{SYSTEM_RULES}\n"
        "Apply the following domain directive if provided:\n"
        "{domain_directive}\n"
        "Use the researcher's output to write a grounded answer. "
        "Include an 'Implications' section. Avoid claims not present in the cited context. "
        "END with the required bold 'Impact on AI:' sentence."
    ),
    expected_output=(
        "A concise, cited answer with sections 'Summary', 'Evidence', 'Implications', "
        "and a final bold 'Impact on AI:' sentence."
    ),
    agent=domain_expert,
    context=[task_gather],
)
