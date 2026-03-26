from forkledger import ForkLedgerEngine

engine = ForkLedgerEngine(".forkledger/demo-store.json")
engine.add_records_from_payload(engine.load_payload_file("examples/sample_forks.json"))

recommendations = engine.recommend(
    current_state={
        "task_type": "research",
        "web_signal": "conflicted",
        "deadline": "tight",
        "source_quality": "mixed",
    },
    constraints={"accuracy_priority": "high"},
)

print("Recommendations:")
for item in recommendations:
    print(item)

print("\nPolicies:")
for policy in engine.policies():
    print(policy)
