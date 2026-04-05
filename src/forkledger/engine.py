"""ForkLedger main engine — v0.3.0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .counterfactual import (
    fill_regret,
    accumulate_regret,
    branch_win_rate,
    confidence_decay_factor,
    normalized_regret,
)
from .models import ForkRecord
from .policy import distill_policies
from .retrieval import rank_records, recommend_branches, embeddings_available
from .storage import JsonForkStore, SqliteForkStore

StoreBackend = Literal["json", "sqlite"]


class ForkLedgerEngine:
    """High-level API for ForkLedger.

    Parameters
    ----------
    store_path : str or Path
    backend : "json" | "sqlite"
    use_embeddings : bool
    embedding_model : str
    confidence_half_life_days : float
        Exponential decay half-life for confidence weighting.
        Default: 60 days. Set to None to disable decay.
    """

    def __init__(
        self,
        store_path: str | Path = ".forkledger/store.json",
        backend: StoreBackend = "json",
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        confidence_half_life_days: float = 60.0,
    ) -> None:
        self.use_embeddings = use_embeddings and embeddings_available()
        self.embedding_model = embedding_model
        self.confidence_half_life_days = confidence_half_life_days

        if backend == "sqlite":
            self.store: JsonForkStore | SqliteForkStore = SqliteForkStore(store_path)
        else:
            self.store = JsonForkStore(store_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_record(self, record: ForkRecord) -> ForkRecord:
        filled = fill_regret(record)
        self.store.append([filled])
        return filled

    def add_records_from_payload(self, payload: list[dict[str, Any]]) -> list[ForkRecord]:
        records = [fill_regret(ForkRecord.from_dict(item)) for item in payload]
        self.store.append(records)
        return records

    def update_outcome(
        self,
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
    ) -> ForkRecord | None:
        return self.store.update_outcome(fork_id, realized_value, confidence)

    def delete(self, fork_id: str) -> bool:
        return self.store.delete(fork_id)

    def purge_expired(self) -> int:
        return self.store.purge_expired()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load(
        self,
        include_expired: bool = False,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> list[ForkRecord]:
        kwargs: dict[str, Any] = {"include_expired": include_expired}
        if isinstance(self.store, SqliteForkStore):
            kwargs["tags"] = tags
            kwargs["min_confidence"] = min_confidence
            kwargs["limit"] = limit
            return self.store.load(**kwargs)
        records = self.store.load(include_expired=include_expired)
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

    # ------------------------------------------------------------------
    # Retrieval & recommendations
    # ------------------------------------------------------------------

    def recommend(
        self,
        current_state: dict[str, Any],
        constraints: dict[str, Any] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        records = self.load(tags=tags, min_confidence=min_confidence)
        return recommend_branches(
            records,
            current_state=current_state,
            constraints=constraints,
            top_k=top_k,
            use_embeddings=self.use_embeddings,
            embedding_model=self.embedding_model,
            min_score=min_score,
        )

    def rank(
        self,
        current_state: dict[str, Any],
        constraints: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[ForkRecord, float]]:
        records = self.load(tags=tags, min_confidence=min_confidence)
        return rank_records(
            records,
            current_state=current_state,
            constraints=constraints,
            use_embeddings=self.use_embeddings,
            embedding_model=self.embedding_model,
        )

    def policies(
        self,
        min_support: int = 2,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        fuzzy: bool = True,
        similarity_threshold: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Distill low-regret policies.

        Args:
            min_support: Minimum records per state pattern.
            fuzzy: Use fuzzy clustering (recommended). Default True.
            similarity_threshold: State overlap threshold for fuzzy mode.
        """
        return distill_policies(
            self.load(tags=tags, min_confidence=min_confidence),
            min_support=min_support,
            fuzzy=fuzzy,
            similarity_threshold=similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Analytics — Tier 2 additions
    # ------------------------------------------------------------------

    def accumulated_regret(
        self,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, float]:
        """Return CFR-style accumulated weighted regret per branch across all records.

        Uses confidence × time-decay weighting.
        Lower = historically better branch.
        """
        records = self.load(tags=tags, min_confidence=min_confidence)
        return accumulate_regret(records, self.confidence_half_life_days)

    def win_rates(
        self,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        """Return win rate per branch (how often each branch was the best choice).

        Returns: {branch: {"wins": int, "appearances": int, "win_rate": float}}
        """
        records = self.load(tags=tags, min_confidence=min_confidence)
        return branch_win_rate(records)

    def decay_factor(self, fork_id: str) -> float | None:
        """Return the current confidence decay factor for a specific fork.

        1.0 = fresh record. Approaches 0.0 as record ages.
        """
        record = self.get(fork_id)
        if record is None:
            return None
        return confidence_decay_factor(record.created_at, self.confidence_half_life_days)

    def audit_trail(
        self,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return a human-readable audit trail of all decisions.

        Sorted by creation date, newest first.
        Includes decay factor for each record.
        """
        records = self.load(tags=tags, limit=limit)
        trail = []
        for r in records:
            decay = confidence_decay_factor(r.created_at, self.confidence_half_life_days)
            trail.append({
                "fork_id":       r.fork_id,
                "created_at":    r.created_at,
                "trigger":       r.trigger,
                "chosen_branch": r.chosen_branch,
                "realized_value":r.realized_value,
                "confidence":    r.confidence,
                "decay_factor":  round(decay, 4),
                "effective_weight": round(r.confidence * decay, 4),
                "regret_vector": r.regret_vector,
                "tags":          r.tags,
            })
        return trail

    # ------------------------------------------------------------------
    # Stats & diagnostics
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
        # Add win rates to stats
        try:
            wr = self.win_rates()
            if wr:
                top_winner = max(wr, key=lambda b: wr[b]["win_rate"])
                s["top_winning_branch"] = {
                    "branch":   top_winner,
                    "win_rate": wr[top_winner]["win_rate"],
                }
        except Exception:
            pass
        return s

    def info(self) -> dict[str, Any]:
        return {
            "backend":                  type(self.store).__name__,
            "use_embeddings":           self.use_embeddings,
            "embedding_model":          self.embedding_model if self.use_embeddings else None,
            "embeddings_package":       embeddings_available(),
            "confidence_half_life_days": self.confidence_half_life_days,
            "version":                  "0.3.0",
        }

    # ------------------------------------------------------------------
    # Utilities
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
