"""
ForkLedger × LangChain — Quickstart
=====================================
Run this to see ForkLedgerMemory in action.

    pip install forkledger[langchain]
    pip install langchain-openai
    export OPENAI_API_KEY=sk-...
    python examples/langchain_quickstart.py
"""

import os, tempfile, uuid
from forkledger.integrations.langchain import ForkLedgerMemory

# --- Setup memory ---
tmp = tempfile.mkdtemp()
memory = ForkLedgerMemory(
    store_path=f"{tmp}/langchain.db",
    backend="sqlite",
    agent_id="demo-agent",
    top_k=3,
    domain="default",
)

print("=== ForkLedger × LangChain Demo ===\n")

# --- Record some past decisions manually ---
print("[1] Recording past decisions...")

memory.record_decision(
    fork_id="lc-001",
    situation={"task": "research", "signal": "mixed", "deadline": "tight"},
    trigger="Sources contradict each other",
    chosen="verify",
    alternatives=["fast-publish", "wait"],
    outcome=1.8,
    estimated_alternatives={"fast-publish": -1.0, "wait": 0.5},
    confidence=0.85,
)
print("  ✓ Recorded: chose 'verify' → outcome=1.8")

memory.record_decision(
    fork_id="lc-002",
    situation={"task": "research", "signal": "mixed", "deadline": "normal"},
    trigger="Ambiguous data quality",
    chosen="verify",
    alternatives=["fast-publish", "skip"],
    outcome=1.5,
    estimated_alternatives={"fast-publish": -0.5, "skip": 0.0},
    confidence=0.9,
)
print("  ✓ Recorded: chose 'verify' → outcome=1.5")

memory.record_decision(
    fork_id="lc-003",
    situation={"task": "research", "signal": "mixed", "deadline": "tight"},
    trigger="Conflicting reports",
    chosen="fast-publish",
    alternatives=["verify", "wait"],
    outcome=-2.0,
    estimated_alternatives={"verify": 1.2, "wait": 0.0},
    confidence=0.7,
)
print("  ✓ Recorded: chose 'fast-publish' → outcome=-2.0")

# --- Update outcome after observation ---
print("\n[2] Updating outcome for lc-001 (it performed better than expected)...")
memory.update_outcome("lc-001", realized_value=2.1)
print("  ✓ Updated lc-001 → realized_value=2.1")

# --- Load memory variables (what the LLM would see) ---
print("\n[3] Loading decision context for a new similar situation...")
context = memory.load_memory_variables({
    "task": "research",
    "signal": "mixed",
    "deadline": "tight",
    "input": "Should I publish this analysis now or verify sources first?",
})
print("\n--- Decision Context injected into LLM ---")
print(context["decision_context"])

# --- Get explicit recommendations ---
print("\n[4] Direct recommendations:")
recs = memory.get_recommendations({
    "task": "research", "signal": "mixed", "deadline": "tight"
})
for i, r in enumerate(recs, 1):
    print(f"  #{i} '{r['branch']}' — score={r['score']:.4f}")

# --- Policies ---
print("\n[5] Distilled policies:")
for p in memory.get_policies():
    print(f"  State={p['state']} → best='{p['recommended_branch']}' (support={p['support']})")

# --- Stats ---
print(f"\n[6] Memory stats: {memory.stats()}")

print("\n=== Done. ForkLedgerMemory is working. ===")

# --- LangChain ConversationChain integration (if langchain-openai installed) ---
if os.getenv("OPENAI_API_KEY"):
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationChain

        memory2 = ForkLedgerMemory(
            store_path=f"{tmp}/langchain2.db",
            backend="sqlite",
            agent_id="openai-agent",
            memory_key="decision_context",
        )

        chain = ConversationChain(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            memory=memory2,
            verbose=False,
        )

        print("\n[7] Running with ConversationChain...")
        response = chain.predict(
            input="Should I verify my research sources or publish quickly?",
            decision_context=memory2.load_memory_variables({
                "task": "research", "signal": "mixed"
            })["decision_context"]
        )
        print(f"Response: {response[:200]}")
    except ImportError:
        print("\n[7] Skipping ConversationChain (langchain-openai not installed)")
