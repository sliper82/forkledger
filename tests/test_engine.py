from forkledger import ForkLedgerEngine
from forkledger.counterfactual import compute_regret
from forkledger.models import Branch, ForkRecord, OutcomeEstimate


def test_compute_regret():
    record = ForkRecord(
        fork_id="x",
        pre_state={"regime": "bullish"},
        trigger="test",
        possible_branches=[Branch(name="a"), Branch(name="b")],
        chosen_branch="a",
        realized_value=-1.0,
        estimated_outcomes=[OutcomeEstimate(branch_name="b", estimated_value=2.0)],
    )
    regret = compute_regret(record)
    assert regret["b"] == 0.0
    assert regret["a"] == 3.0


def test_recommend_and_policy(tmp_path):
    store = tmp_path / "store.json"
    engine = ForkLedgerEngine(store)
    payload = [
        {
            "fork_id": "f1",
            "pre_state": {"task": "research", "signal": "mixed"},
            "trigger": "x",
            "possible_branches": [{"name": "fast"}, {"name": "verify"}],
            "chosen_branch": "fast",
            "realized_value": -1.0,
            "estimated_outcomes": [{"branch_name": "verify", "estimated_value": 1.0}],
        },
        {
            "fork_id": "f2",
            "pre_state": {"task": "research", "signal": "mixed"},
            "trigger": "y",
            "possible_branches": [{"name": "fast"}, {"name": "verify"}],
            "chosen_branch": "verify",
            "realized_value": 0.8,
            "estimated_outcomes": [{"branch_name": "fast", "estimated_value": -0.5}],
        },
    ]
    engine.add_records_from_payload(payload)

    recommendations = engine.recommend({"task": "research", "signal": "mixed"})
    assert recommendations[0]["branch"] == "verify"

    policies = engine.policies()
    assert policies[0]["recommended_branch"] == "verify"
