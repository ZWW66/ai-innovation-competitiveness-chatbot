# crew/main.py
import sys
from pathlib import Path

# Support both `python -m crew.main` and direct `python crew/main.py` execution.
if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from crewai import Crew

from crew.tasks import task_answer, task_gather


def kickoff_query(query: str, domain_directive: str):
    crew = Crew(
        agents=[task_gather.agent, task_answer.agent],
        tasks=[task_gather, task_answer],
        verbose=True,
    )
    return crew.kickoff(inputs={
        "query": query,
        "domain_directive": domain_directive,
    })

if __name__ == "__main__":
    from crew.tasks import DOMAIN_DIRECTIVES

    q = "How do recent AI chip export controls affect global AI competitiveness?"
    ans = kickoff_query(q, DOMAIN_DIRECTIVES["general"])
    print("\n=== FINAL ANSWER ===\n")
    print(ans)
