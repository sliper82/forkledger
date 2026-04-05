"""ForkLedger test suite — comprehensive coverage of all upgraded features."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forkledger import ForkLedgerEngine
from forkledger.counterfactual import compute_regret
from forkledger.models import Branch, ForkRecord, OutcomeEstimate
from forkledger.retrieval import embeddings_available
from forkledger.storage import JsonForkStore, SqliteForkStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = [
    {
        "fork_id": "f1",
        "pre_state": {"task": "research", "signal": "mixed"},
        "trigger": "conflicting data",
        "possible_branches": [{"name": "fast"}, {"name": "verify"}],
        "chosen_branch": "fast",
        "realized_value": -1.0,
        "estimated_outcomes": [{"branch_name": "verify", "estimated_value": 1.0}],
        "tags": ["research"],
        "confidence": 0.7,
    },
    {
        "fork_id": "f2",
        "pre_state": {"task": "research", "signal": "mixed"},
        "trigger": "ambiguous result",
        "possible_branches": [{"name": "fast"}, {"name": "verify"}],
        "chosen_branch": "verify",
        "realized_value": 0.8,
        "estimated_outcomes": [{"branch_name": "fast", "estimated_value": -0.5}],
        "tags": ["research"],
        "confidence": 0.9,
    },
    {
        "fork_id": "f3",
        "pre_state": {"task": "trade", "signal": "bullish"},
        "trigger": "breakout signal",
        "possible_branches": [{"name": "enter"}, {"name": "wait"}],
        "chosen_branch": "enter",
        "realized_value": 2.5,
        "estimated_outcomes": [{"branch_name": "wait", "estimated_value": 0.0}],
        "tags": ["crypto", "trade"],
        "confidence": 0.8,
    },
]


@pytest.fixture(params=["json", "sqlite"])
def engine(tmp_path, request):
    backend = request.param
    if backend == "json":
        store_path = tmp_path / "store.json"
    else:
        store_path = tmp_path / "store.db"
    eng = ForkLedgerEngine(store_path=store_path, backend=backend)
    eng.add_records_from_payload(SAMPLE_PAYLOAD)
    return eng


# ---------------------------------------------------------------------------
# Counterfactual regret
# ---------------------------------------------------------------------------

class TestRegret:
    def test_chosen_has_nonzero_regret_when_worse(self):
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

    def test_best_branch_has_zero_regret(self):
        record = ForkRecord(
            fork_id="y",
            pre_state={},
            trigger="t",
            possible_branches=[Branch(name="win"), Branch(name="lose")],
            chosen_branch="win",
            realized_value=5.0,
            estimated_outcomes=[OutcomeEstimate(branch_name="lose", estimated_value=1.0)],
        )
        regret = compute_regret(record)
        assert regret["win"] == 0.0
        assert regret["lose"] == 4.0

    def test_empty_outcomes_returns_only_chosen(self):
        record = ForkRecord(
            fork_id="z",
            pre_state={},
            trigger="t",
            possible_branches=[Branch(name="a")],
            chosen_branch="a",
            realized_value=1.0,
        )
        regret = compute_regret(record)
        assert regret["a"] == 0.0


# ---------------------------------------------------------------------------
# Recommendation & policy
# ---------------------------------------------------------------------------

class TestRecommendAndPolicy:
    def test_recommend_returns_verify_first(self, engine):
        recs = engine.recommend({"task": "research", "signal": "mixed"})
        assert recs[0]["branch"] == "verify"

    def test_policy_recommends_verify(self, engine):
        policies = engine.policies()
        research_policy = next(p for p in policies if p["state"].get("task") == "research")
        assert research_policy["recommended_branch"] == "verify"

    def test_min_score_filter(self, engine):
        # A completely different state should return nothing above 0.9
        recs = engine.recommend({"task": "unrelated", "signal": "none"}, min_score=0.9)
        assert recs == []

    def test_tag_filter(self, engine):
        records = engine.load(tags=["crypto"])
        assert all("crypto" in r.tags for r in records)
        assert len(records) == 1

    def test_min_confidence_filter(self, engine):
        records = engine.load(min_confidence=0.85)
        assert all(r.confidence >= 0.85 for r in records)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_get_existing(self, engine):
        r = engine.get("f1")
        assert r is not None
        assert r.fork_id == "f1"

    def test_get_nonexistent(self, engine):
        assert engine.get("does-not-exist") is None

    def test_update_outcome_recomputes_regret(self, engine):
        updated = engine.update_outcome("f1", realized_value=3.0, confidence=0.95)
        assert updated is not None
        assert updated.realized_value == 3.0
        assert updated.confidence == 0.95
        # After update realized_value=3.0 > estimated_value=1.0 → chosen has 0 regret
        assert updated.regret_vector.get("fast", -1) == 0.0

    def test_update_outcome_not_found(self, engine):
        result = engine.update_outcome("ghost", 1.0)
        assert result is None

    def test_delete_removes_record(self, engine):
        assert engine.delete("f3") is True
        assert engine.get("f3") is None

    def test_delete_nonexistent_returns_false(self, engine):
        assert engine.delete("nope") is False

    def test_deduplication_on_append(self, engine):
        before = len(engine.load())
        # Re-add same fork_id with different value
        engine.add_records_from_payload([{**SAMPLE_PAYLOAD[0], "realized_value": 99.0}])
        after = len(engine.load())
        assert before == after  # count unchanged
        r = engine.get("f1")
        assert r.realized_value == 99.0


# ---------------------------------------------------------------------------
# Expiry enforcement
# ---------------------------------------------------------------------------

class TestExpiry:
    def test_expired_record_excluded_by_default(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "exp.json", backend="json")
        payload = [{
            **SAMPLE_PAYLOAD[0],
            "fork_id": "expired-fork",
            "expiry": "2000-01-01T00:00:00+00:00",  # clearly in the past
        }]
        eng.add_records_from_payload(payload)
        records = eng.load(include_expired=False)
        assert all(r.fork_id != "expired-fork" for r in records)

    def test_expired_record_included_when_requested(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "exp2.json", backend="json")
        payload = [{
            **SAMPLE_PAYLOAD[0],
            "fork_id": "expired-fork2",
            "expiry": "2000-01-01T00:00:00+00:00",
        }]
        eng.add_records_from_payload(payload)
        records = eng.load(include_expired=True)
        assert any(r.fork_id == "expired-fork2" for r in records)

    def test_purge_expired(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "purge.json", backend="json")
        payload = [
            {**SAMPLE_PAYLOAD[0], "fork_id": "live"},
            {**SAMPLE_PAYLOAD[1], "fork_id": "dead", "expiry": "2000-01-01T00:00:00+00:00"},
        ]
        eng.add_records_from_payload(payload)
        removed = eng.purge_expired()
        assert removed == 1
        assert eng.get("live") is not None


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_and_reimport(self, tmp_path, engine):
        out = tmp_path / "export.json"
        engine.export_json(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 3

        # Import into a fresh engine
        fresh = ForkLedgerEngine(tmp_path / "fresh.json", backend="json")
        count = fresh.import_json(out)
        assert count == 3


# ---------------------------------------------------------------------------
# Stats & info
# ---------------------------------------------------------------------------

class TestStatsInfo:
    def test_stats_returns_totals(self, engine):
        s = engine.stats()
        assert s["total_forks"] == 3

    def test_info_returns_backend(self, engine):
        i = engine.info()
        assert "backend" in i
        assert "use_embeddings" in i


# ---------------------------------------------------------------------------
# Embeddings (skipped if not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not embeddings_available(), reason="sentence-transformers not installed")
class TestEmbeddings:
    def test_embedding_recommend_runs(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "emb.json", backend="json", use_embeddings=True)
        eng.add_records_from_payload(SAMPLE_PAYLOAD)
        recs = eng.recommend({"task": "research", "signal": "mixed"})
        assert len(recs) > 0


# ---------------------------------------------------------------------------
# v0.4.0 — New features
# ---------------------------------------------------------------------------

class TestNewModels:
    def test_updated_at_exists(self, engine):
        r = engine.get("f1")
        assert hasattr(r, "updated_at")
        assert r.updated_at is not None

    def test_namespace_default(self, engine):
        r = engine.get("f1")
        assert r.namespace == "default"

    def test_outcome_source_default(self, engine):
        r = engine.get("f1")
        assert r.outcome_source == "observed"

    def test_touch_updates_updated_at(self, tmp_path):
        import time
        eng = ForkLedgerEngine(tmp_path / "touch.json", backend="json")
        eng.add_records_from_payload(SAMPLE_PAYLOAD)
        r_before = eng.get("f1")
        t_before = r_before.updated_at
        time.sleep(0.05)
        eng.update_outcome("f1", 99.0)
        r_after = eng.get("f1")
        assert r_after.updated_at > t_before

    def test_outcome_source_preserved(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "src.json", backend="json")
        eng.add_records_from_payload(SAMPLE_PAYLOAD)
        r = eng.update_outcome("f1", 1.0, outcome_source="simulated")
        assert r.outcome_source == "simulated"


class TestNumericSimilarity:
    def test_numeric_close_values_score_high(self, tmp_path):
        """Records with numerically close states should score higher than very different ones."""
        from forkledger.retrieval import _state_overlap
        close = _state_overlap({"price": 100}, {"price": 101})
        far   = _state_overlap({"price": 100}, {"price": 1000})
        assert close > far
        assert close > 0.9

    def test_exact_numeric_match(self):
        from forkledger.retrieval import _state_overlap
        s = _state_overlap({"x": 5.0}, {"x": 5.0})
        assert s == 1.0

    def test_mixed_state(self):
        from forkledger.retrieval import _state_overlap
        s = _state_overlap(
            {"task": "trade", "signal": "bullish", "price": 100},
            {"task": "trade", "signal": "bullish", "price": 102},
        )
        assert s > 0.9  # mostly matching with small numeric diff


class TestConstraintScoringFix:
    def test_empty_constraints_scores_neutral(self):
        from forkledger.retrieval import _constraint_overlap
        # FIXED: empty query constraints → 0.5 (neutral), not 1.0
        score = _constraint_overlap({"risk": "low"}, {})
        assert score == 0.5

    def test_matching_constraints_scores_high(self):
        from forkledger.retrieval import _constraint_overlap
        score = _constraint_overlap({"risk": "low"}, {"risk": "low"})
        assert score == 1.0

    def test_empty_record_constraints_scores_zero(self):
        from forkledger.retrieval import _constraint_overlap
        score = _constraint_overlap({}, {"risk": "low"})
        assert score == 0.0


class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        from forkledger.retrieval import ScoringWeights
        import math
        w = ScoringWeights()
        total = (w.state_similarity + w.constraint_match +
                 w.recency + w.confidence + w.regret_salience)
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_domain_preset_trading(self):
        from forkledger.retrieval import ScoringWeights
        w = ScoringWeights.for_domain("trading")
        assert w.recency > ScoringWeights().recency  # trading weights recency more

    def test_invalid_weights_raises(self):
        from forkledger.retrieval import ScoringWeights
        import pytest
        with pytest.raises(ValueError):
            ScoringWeights(state_similarity=0.9)  # doesn't sum to 1.0


class TestNamespaceIsolation:
    def test_namespace_filter(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "ns.db", backend="sqlite")
        p1 = [{**SAMPLE_PAYLOAD[0], "fork_id": "ns-a1", "namespace": "agent_a"}]
        p2 = [{**SAMPLE_PAYLOAD[1], "fork_id": "ns-b1", "namespace": "agent_b"}]
        eng.add_records_from_payload(p1 + p2)
        a_records = eng.load(namespace="agent_a")
        b_records = eng.load(namespace="agent_b")
        assert all(r.namespace == "agent_a" for r in a_records)
        assert all(r.namespace == "agent_b" for r in b_records)
        assert len(a_records) == 1
        assert len(b_records) == 1


class TestFTSSearch:
    def test_fts_search_finds_by_trigger(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "fts.db", backend="sqlite")
        eng.add_records_from_payload(SAMPLE_PAYLOAD)
        results = eng.search("conflicting", limit=10)
        # f1 has trigger "conflicting data"
        assert any(r.fork_id == "f1" for r in results)

    def test_fts_search_json_fallback(self, tmp_path):
        eng = ForkLedgerEngine(tmp_path / "fts.json", backend="json")
        eng.add_records_from_payload(SAMPLE_PAYLOAD)
        results = eng.search("conflicting", limit=10)
        assert isinstance(results, list)


class TestAuditTrail:
    def test_audit_has_updated_at(self, engine):
        trail = engine.audit_trail()
        assert all("updated_at" in e for e in trail)

    def test_audit_has_outcome_source(self, engine):
        trail = engine.audit_trail()
        assert all("outcome_source" in e for e in trail)

    def test_audit_has_namespace(self, engine):
        trail = engine.audit_trail()
        assert all("namespace" in e for e in trail)
