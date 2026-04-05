"""
ForkLedger Quickstart
=====================
Run this file to see ForkLedger in action in under 60 seconds.

    pip install forkledger
    python examples/quickstart.py
"""

import tempfile, os
from forkledger import ForkLedgerEngine, ForkRecord, Branch, OutcomeEstimate

# Use a temp store so this script is self-contained
tmp = tempfile.mkdtemp()
engine = ForkLedgerEngine(os.path.join(tmp, "quickstart.db"), backend="sqlite")

print("=" * 60)
print("ForkLedger Quickstart")
print("=" * 60)

# ── Step 1: Record some past decisions ───────────────────────────────────────
print("\n[1] Recording past decisions...")

decisions = [
    ForkRecord(
        fork_id="research-001",
        pre_state={"task": "research", "signal": "conflicted", "deadline": "tight"},
        trigger="Sources contradict each other — act or verify?",
        possible_branches=[Branch("verify"), Branch("fast-publish"), Branch("wait")],
        chosen_branch="fast-publish",
        realized_value=-2.0,
        estimated_outcomes=[
            OutcomeEstimate("verify",       estimated_value=1.5, confidence=0.7),
            OutcomeEstimate("wait",         estimated_value=0.0, confidence=0.5),
        ],
        confidence=0.6,
        tags=["research", "decision"],
    ),
    ForkRecord(
        fork_id="research-002",
        pre_state={"task": "research", "signal": "conflicted", "deadline": "normal"},
        trigger="Same contradiction, more time available",
        possible_branches=[Branch("verify"), Branch("fast-publish"), Branch("wait")],
        chosen_branch="verify",
        realized_value=1.8,
        estimated_outcomes=[
            OutcomeEstimate("fast-publish", estimated_value=-1.0, confidence=0.7),
            OutcomeEstimate("wait",         estimated_value=0.5,  confidence=0.5),
        ],
        confidence=0.85,
        tags=["research", "decision"],
    ),
    ForkRecord(
        fork_id="research-003",
        pre_state={"task": "research", "signal": "conflicted", "deadline": "tight"},
        trigger="Conflicting data again, tight window",
        possible_branches=[Branch("verify"), Branch("fast-publish"), Branch("wait")],
        chosen_branch="verify",
        realized_value=1.2,
        estimated_outcomes=[
            OutcomeEstimate("fast-publish", estimated_value=-1.5, confidence=0.75),
        ],
        confidence=0.8,
        tags=["research"],
    ),
]

for d in decisions:
    engine.add_record(d)
    print(f"  ✓ {d.fork_id} → chose '{d.chosen_branch}' → value={d.realized_value}")

# ── Step 2: Update an outcome after the fact ─────────────────────────────────
print("\n[2] Updating outcome for research-001 (realized it was even worse)...")
updated = engine.update_outcome("research-001", realized_value=-3.0, confidence=0.9)
print(f"  ✓ research-001 regret: {updated.regret_vector}")

# ── Step 3: Ask for recommendations ──────────────────────────────────────────
print("\n[3] Current situation: conflicted signal, tight deadline.")
print("    What does ForkLedger recommend?\n")

recs = engine.recommend(
    current_state={"task": "research", "signal": "conflicted", "deadline": "tight"},
    top_k=3,
)

for i, r in enumerate(recs, 1):
    support_ids = [s["fork_id"] for s in r["support"]]
    print(f"  #{i}  branch='{r['branch']}'  score={r['score']}  evidence={support_ids}")

# ── Step 4: Distilled policies ────────────────────────────────────────────────
print("\n[4] Distilled policies (min 2 supporting forks):\n")
for p in engine.policies(min_support=2):
    print(f"  state={p['state']}")
    print(f"  → recommended: '{p['recommended_branch']}'  (support={p['support']})")
    print(f"  → avg regret:  {p['average_regret']}\n")

# ── Step 5: Stats ─────────────────────────────────────────────────────────────
print("[5] Store stats:")
for k, v in engine.stats().items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("Done. ForkLedger is working.")
print(f"Store: {tmp}/quickstart.db")
print("=" * 60)
