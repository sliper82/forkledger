"""
ForkLedger × AutoGen — Quickstart
====================================
Run this to see ForkLedgerHook in action.

    pip install forkledger[autogen]
    pip install pyautogen
    python examples/autogen_quickstart.py

No API key required — runs in offline demo mode.
"""

import tempfile
from forkledger.integrations.autogen import ForkLedgerHook, autogen_available

print("=== ForkLedger × AutoGen Demo ===\n")

# --- Setup hook ---
tmp = tempfile.mkdtemp()
hook = ForkLedgerHook(
    store_path=f"{tmp}/autogen.db",
    backend="sqlite",
    agent_id="research-agent",
    top_k=3,
    domain="research",
    inject_context=True,
)

# --- Record past decisions ---
print("[1] Recording past agent decisions...")

hook.record(
    situation={"task": "web_research", "query_type": "factual", "sources": "multiple"},
    trigger="Multiple conflicting sources found",
    chosen="verify_cross_reference",
    alternatives=["use_first_source", "ask_human", "skip"],
    outcome=1.8,
    estimated={"use_first_source": -0.5, "ask_human": 0.3, "skip": -1.0},
    confidence=0.85,
    tags=["research", "web"],
)
print("  ✓ Recorded: chose 'verify_cross_reference' → outcome=1.8")

hook.record(
    situation={"task": "web_research", "query_type": "factual", "sources": "single"},
    trigger="Only one source available",
    chosen="use_first_source",
    alternatives=["verify_cross_reference", "skip"],
    outcome=0.5,
    estimated={"verify_cross_reference": 1.0, "skip": -0.5},
    confidence=0.75,
    tags=["research", "web"],
)
print("  ✓ Recorded: chose 'use_first_source' → outcome=0.5")

hook.record(
    situation={"task": "web_research", "query_type": "factual", "sources": "multiple"},
    trigger="Contradicting data from 3 sources",
    chosen="verify_cross_reference",
    alternatives=["use_first_source", "ask_human"],
    outcome=2.1,
    estimated={"use_first_source": -1.0, "ask_human": 0.8},
    confidence=0.9,
)
print("  ✓ Recorded: chose 'verify_cross_reference' → outcome=2.1")

# --- Get recommendations for new situation ---
print("\n[2] Recommendations for new situation:")
recs = hook.recommend({
    "task": "web_research", "query_type": "factual", "sources": "multiple"
})
for i, r in enumerate(recs, 1):
    support = len(r["support"])
    print(f"  #{i} '{r['branch']}' — score={r['score']:.4f} (supported by {support} past decisions)")

# --- Distilled policies ---
print("\n[3] Distilled policies:")
for p in hook.policies():
    print(f"  State={p['state']}")
    print(f"  → Best: '{p['recommended_branch']}' | Win rate: {p.get('win_rate', {})}")

# --- Context injection demo ---
print("\n[4] Context injection demo:")
hook._last_state = {"task": "web_research", "query_type": "factual", "sources": "multiple"}
test_message = "I found multiple conflicting sources. What should I do?"
injected = hook._before_send(test_message)
print(f"Original: '{test_message}'")
print(f"Injected:\n{injected}")

# --- AutoGen integration (if installed) ---
print("\n[5] AutoGen agent attachment:")
if autogen_available():
    try:
        from autogen import AssistantAgent
        assistant = AssistantAgent(
            "research_assistant",
            llm_config={"config_list": [{"model": "gpt-4", "api_key": "fake"}]},
            system_message="You are a research assistant.",
        )
        hook.attach(assistant)
        print("  ✓ Hook attached to AutoGen AssistantAgent")
    except Exception as e:
        print(f"  ⚠ Could not attach to agent: {e}")
else:
    print("  ℹ AutoGen not installed (pip install pyautogen)")
    print("  ✓ ForkLedgerHook works standalone without AutoGen")

print(f"\n[6] Stats: {hook.stats()}")
print("\n=== Done. ForkLedgerHook is working. ===")
