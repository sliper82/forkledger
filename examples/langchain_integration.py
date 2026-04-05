"""
ForkLedger × LangChain Integration Example
==========================================
Shows how to use ForkLedger as a decision memory layer
alongside a LangChain agent.

This is a standalone pattern — no LangChain install required to read it.
"""

# Pattern: wrap ForkLedger as a LangChain Tool

INTEGRATION_PATTERN = '''
from langchain.tools import tool
from forkledger import ForkLedgerEngine, ForkRecord, Branch, OutcomeEstimate
import uuid

engine = ForkLedgerEngine("agent_memory.db", backend="sqlite")

@tool
def remember_decision(
    situation: str,
    chosen: str,
    alternatives: list[str],
    outcome: float,
    tags: list[str] = []
) -> str:
    """Record a decision the agent just made, with its outcome."""
    record = ForkRecord(
        fork_id=str(uuid.uuid4()),
        pre_state={"situation": situation},
        trigger=situation,
        possible_branches=[Branch(name=a) for a in [chosen] + alternatives],
        chosen_branch=chosen,
        realized_value=outcome,
        tags=tags,
    )
    engine.add_record(record)
    return f"Recorded decision: chose '{chosen}' with value {outcome}"

@tool
def recall_best_action(situation: str, top_k: int = 3) -> str:
    """Given a situation, recall what actions historically worked best."""
    recs = engine.recommend(
        current_state={"situation": situation},
        top_k=top_k,
    )
    if not recs:
        return "No historical data for this situation."
    lines = [f"- '{r['branch']}' (score={r['score']})" for r in recs]
    return "Recommended actions based on history:\\n" + "\\n".join(lines)

@tool
def get_decision_patterns() -> str:
    """Return distilled patterns from repeated similar decisions."""
    policies = engine.policies(min_support=2)
    if not policies:
        return "Not enough data yet to extract patterns."
    lines = []
    for p in policies:
        lines.append(
            f"State {p['state']} → best action: '{p['recommended_branch']}' "
            f"(seen {p['support']} times)"
        )
    return "\\n".join(lines)

# Usage in your agent:
# tools = [remember_decision, recall_best_action, get_decision_patterns]
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
'''

print("LangChain integration pattern:")
print(INTEGRATION_PATTERN)
