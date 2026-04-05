"""ForkLedger engine — v0.4.0"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .counterfactual import (
    fill_regret, accumulate_regret, branch_win_rate,
    confidence_decay_factor,
)
from .models import ForkRecord
from .policy import distill_policies
from .retrieval import (
    rank_records, recommend_branches, embeddings_available,
    faiss_available, ScoringWeights, FaissIndex,
)
from .storage import JsonForkStore, SqliteForkStore

StoreBackend = Literal["json", "sqlite"]


class ForkLedgerEngine:
    """High-level API for ForkLedger v0.4.0.

    Parameters
    ----------
    store_path : str or Path
    backend : "json" | "sqlite"
    use_embeddings : bool
    embedding_model : str
    confidence_half_life_days : float
    use_faiss : bool
        Use FAISS HNSW index for O(log n) vector search.
        Requires: pip install forkledger[faiss]
    """

    def __init__(
        self,
        store_path: str | Path = ".forkledger/store.json",
        backend: StoreBackend = "json",
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        confidence_half_life_days: float = 60.0,
        use_faiss: bool = False,
    ) -> None:
        self.use_embeddings = use_embeddings and embeddings_available()
        self.embedding_model = embedding_model
        self.confidence_half_life_days = confidence_half_life_days
        self.use_faiss = use_faiss and faiss_available() and self.use_embeddings
        self._faiss_index: FaissIndex | None = FaissIndex() if self.use_faiss else None

        if backend == "sqlite":
            self.store: JsonForkStore | SqliteForkStore = SqliteForkStore(store_path)
        else:
            self.store = JsonForkStore(store_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_record(self, record: ForkRecord) -> ForkRecord:
        filled = fill_regret(record)
        self.store.append([filled])
        self._invalidate_faiss()
        return filled

    def add_records_from_payload(self, payload: list[dict[str, Any]]) -> list[ForkRecord]:
        records = [fill_regret(ForkRecord.from_dict(item)) for item in payload]
        self.store.append(records)
        self._invalidate_faiss()
        return records

    def update_outcome(
        self,
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
        outcome_source: str = "observed",
    ) -> ForkRecord | None:
        return self.store.update_outcome(fork_id, realized_value, confidence, outcome_source)

    def delete(self, fork_id: str) -> bool:
        ok = self.store.delete(fork_id)
        if ok:
            self._invalidate_faiss()
        return ok

    def purge_expired(self) -> int:
        count = self.store.purge_expired()
        if count:
            self._invalidate_faiss()
        return count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(
        self,
        include_expired: bool = False,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
        namespace: str | None = None,
        full_text_query: str | None = None,
    ) -> list[ForkRecord]:
        kwargs: dict[str, Any] = {"include_expired": include_expired}
        if isinstance(self.store, SqliteForkStore):
            kwargs.update(
                tags=tags, min_confidence=min_confidence,
                limit=limit, namespace=namespace,
                full_text_query=full_text_query,
            )
            return self.store.load(**kwargs)
        # JSON store
        records = self.store.load(include_expired=include_expired, namespace=namespace)
        if min_confidence > 0:
            records = [r for r in records if r.confidence >= min_confidence]
        if tags:
            tag_set = set(tags)
            records = [r for r in records if tag_set.intersection(r.tags)]
        if limit:
            records = records[:limit]
        return records

    def get(self, fork_id: str) -> ForkRecord | None:
        if isinstance(self.store, SqliteForkStore):
            return self.store.get(fork_id)
        records = {r.fork_id: r for r in self.store.load(include_expired=True)}
        return records.get(fork_id)

    def search(self, query: str, limit: int = 20) -> list[ForkRecord]:
        """Full-text search across trigger, state, branch, tags (SQLite FTS5)."""
        if isinstance(self.store, SqliteForkStore):
            return self.store.search_text(query, limit=limit)
        # JSON fallback: simple substring match
        records = self.store.load()
        q = query.lower()
        return [
            r for r in records
            if q in r.trigger.lower()
            or q in r.chosen_branch.lower()
            or any(q in t.lower() for t in r.tags)
        ][:limit]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recommend(
        self,
        current_state: dict[str, Any],
        constraints: dict[str, Any] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        namespace: str | None = None,
        weights: ScoringWeights | None = None,
        domain: str = "default",
    ) -> list[dict[str, Any]]:
        w = weights or ScoringWeights.for_domain(domain)
        records = self._load_for_retrieval(tags, min_confidence, namespace, current_state, top_k)
        return recommend_branches(
            records, current_state=current_state, constraints=constraints,
            top_k=top_k, use_embeddings=self.use_embeddings,
            embedding_model=self.embedding_model, min_score=min_score,
            weights=w, namespace=None,  # already filtered above
        )

    def rank(
        self,
        current_state: dict[str, Any],
        constraints: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        namespace: str | None = None,
        weights: ScoringWeights | None = None,
        domain: str = "default",
    ) -> list[tuple[ForkRecord, float]]:
        w = weights or ScoringWeights.for_domain(domain)
        records = self.load(tags=tags, min_confidence=min_confidence, namespace=namespace)
        return rank_records(
            records, current_state=current_state, constraints=constraints,
            use_embeddings=self.use_embeddings, embedding_model=self.embedding_model,
            weights=w, namespace=None,
        )

    def _load_for_retrieval(
        self,
        tags: list[str] | None,
        min_confidence: float,
        namespace: str | None,
        current_state: dict[str, Any],
        top_k: int,
    ) -> list[ForkRecord]:
        """Smart loading: use FAISS pre-filtering if available."""
        if self.use_faiss and self._faiss_index and self._faiss_index.available:
            self._ensure_faiss_built()
            candidate_ids = set(
                self._faiss_index.search(current_state, k=top_k * 10,
                                         model=self.embedding_model)
            )
            all_records = self.load(tags=tags, min_confidence=min_confidence,
                                    namespace=namespace)
            # Return candidates + a sample of others for diversity
            candidates = [r for r in all_records if r.fork_id in candidate_ids]
            others = [r for r in all_records if r.fork_id not in candidate_ids]
            return candidates + others[:top_k]
        return self.load(tags=tags, min_confidence=min_confidence, namespace=namespace)

    def _ensure_faiss_built(self) -> None:
        if self._faiss_index and not self._faiss_index._built:
            records = self.load()
            self._faiss_index.build(records, model=self.embedding_model)

    def _invalidate_faiss(self) -> None:
        if self._faiss_index:
            self._faiss_index._built = False

    def policies(
        self,
        min_support: int = 2,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        fuzzy: bool = True,
        similarity_threshold: float = 0.6,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        return distill_policies(
            self.load(tags=tags, min_confidence=min_confidence, namespace=namespace),
            min_support=min_support, fuzzy=fuzzy,
            similarity_threshold=similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def accumulated_regret(
        self,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        namespace: str | None = None,
    ) -> dict[str, float]:
        records = self.load(tags=tags, min_confidence=min_confidence, namespace=namespace)
        return accumulate_regret(records, self.confidence_half_life_days)

    def win_rates(
        self,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        namespace: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        records = self.load(tags=tags, min_confidence=min_confidence, namespace=namespace)
        return branch_win_rate(records)

    def decay_factor(self, fork_id: str) -> float | None:
        record = self.get(fork_id)
        if record is None:
            return None
        return confidence_decay_factor(record.created_at, self.confidence_half_life_days)

    def audit_trail(
        self,
        tags: list[str] | None = None,
        limit: int = 50,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        records = self.load(tags=tags, limit=limit, namespace=namespace)
        trail = []
        for r in records:
            decay = confidence_decay_factor(r.created_at, self.confidence_half_life_days)
            trail.append({
                "fork_id":        r.fork_id,
                "created_at":     r.created_at,
                "updated_at":     r.updated_at,
                "trigger":        r.trigger,
                "chosen_branch":  r.chosen_branch,
                "realized_value": r.realized_value,
                "outcome_source": r.outcome_source,
                "confidence":     r.confidence,
                "decay_factor":   round(decay, 4),
                "effective_weight": round(r.confidence * decay, 4),
                "regret_vector":  r.regret_vector,
                "tags":           r.tags,
                "namespace":      r.namespace,
            })
        return trail

    # ------------------------------------------------------------------
    # Stats & info
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        if isinstance(self.store, SqliteForkStore):
            s = self.store.stats()
        else:
            records = self.load(include_expired=True)
            s = {
                "total_forks":    len(records),
                "avg_confidence": round(
                    sum(r.confidence for r in records) / max(len(records), 1), 4
                ),
            }
        try:
            wr = self.win_rates()
            if wr:
                top = max(wr, key=lambda b: wr[b]["win_rate"])
                s["top_winning_branch"] = {
                    "branch":   top,
                    "win_rate": wr[top]["win_rate"],
                }
        except Exception:
            pass
        return s

    def info(self) -> dict[str, Any]:
        return {
            "backend":                   type(self.store).__name__,
            "use_embeddings":            self.use_embeddings,
            "embedding_model":           self.embedding_model if self.use_embeddings else None,
            "embeddings_package":        embeddings_available(),
            "faiss_enabled":             self.use_faiss,
            "faiss_available":           faiss_available(),
            "confidence_half_life_days": self.confidence_half_life_days,
            "version":                   "0.4.0",
        }

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    @staticmethod
    def load_payload_file(path: str | Path) -> list[dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def export_json(self, path: str | Path) -> None:
        records = self.load(include_expired=True)
        payload = [r.to_dict() for r in records]
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def import_json(self, path: str | Path) -> int:
        payload = self.load_payload_file(path)
        records = self.add_records_from_payload(payload)
        return len(records)
