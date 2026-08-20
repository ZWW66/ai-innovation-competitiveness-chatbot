# crew/main.py
from crewai import Crew
from crew.tasks import task_gather, task_answer

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
