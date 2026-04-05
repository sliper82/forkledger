"""ForkLedger main engine.

ForkLedgerEngine is the single entry point for all operations.
Supports both JsonForkStore (default) and SqliteForkStore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .counterfactual import fill_regret
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
        Path to the storage file (.json or .db).
    backend : "json" | "sqlite"
        Storage backend. Default is "json" for backward compatibility.
        Use "sqlite" for any serious workload.
    use_embeddings : bool
        Enable semantic similarity via sentence-transformers.
        Requires: pip install forkledger[embeddings]
    embedding_model : str
        sentence-transformers model name. Default: "all-MiniLM-L6-v2"
    """

    def __init__(
        self,
        store_path: str | Path = ".forkledger/store.json",
        backend: StoreBackend = "json",
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.use_embeddings = use_embeddings and embeddings_available()
        self.embedding_model = embedding_model

        if backend == "sqlite":
            self.store: JsonForkStore | SqliteForkStore = SqliteForkStore(store_path)
        else:
            self.store = JsonForkStore(store_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_record(self, record: ForkRecord) -> ForkRecord:
        """Add a single ForkRecord. Regret is computed automatically."""
        filled = fill_regret(record)
        self.store.append([filled])
        return filled

    def add_records_from_payload(self, payload: list[dict[str, Any]]) -> list[ForkRecord]:
        """Bulk-add from a list of raw dicts (e.g. loaded from JSON)."""
        records = [fill_regret(ForkRecord.from_dict(item)) for item in payload]
        self.store.append(records)
        return records

    def update_outcome(
        self,
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
    ) -> ForkRecord | None:
        """Update the realized value of an existing fork and recompute regret.

        Returns the updated record, or None if fork_id not found.
        """
        return self.store.update_outcome(fork_id, realized_value, confidence)

    def delete(self, fork_id: str) -> bool:
        """Delete a fork by ID. Returns True if deleted, False if not found."""
        return self.store.delete(fork_id)

    def purge_expired(self) -> int:
        """Remove all expired forks. Returns number of removed records."""
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
        """Load records with optional filters."""
        kwargs: dict[str, Any] = {"include_expired": include_expired}

        # SqliteForkStore supports extra filters natively
        if isinstance(self.store, SqliteForkStore):
            kwargs["tags"] = tags
            kwargs["min_confidence"] = min_confidence
            kwargs["limit"] = limit
            return self.store.load(**kwargs)

        # JsonForkStore: apply filters in Python
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
        """Fetch a single fork by ID."""
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
        """Recommend branches for a given state.

        Parameters
        ----------
        current_state : dict
            Representation of the current decision context.
        constraints : dict, optional
            Active hard constraints.
        top_k : int
            Number of top historical forks to consider.
        min_score : float
            Minimum match score threshold (0.0–1.0).
        tags : list[str], optional
            Filter source records by tags.
        min_confidence : float
            Minimum confidence threshold for source records.
        """
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
        """Rank all stored forks by similarity to current_state."""
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
    ) -> list[dict[str, Any]]:
        """Distill low-regret policies from repeated decision states."""
        return distill_policies(
            self.load(tags=tags, min_confidence=min_confidence),
            min_support=min_support,
        )

    # ------------------------------------------------------------------
    # Stats & diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return store statistics. Richer output for SQLite backend."""
        if isinstance(self.store, SqliteForkStore):
            return self.store.stats()
        records = self.load(include_expired=True)
        return {
            "total_forks":    len(records),
            "avg_confidence": round(sum(r.confidence for r in records) / max(len(records), 1), 4),
        }

    def info(self) -> dict[str, Any]:
        """Return engine configuration info."""
        return {
            "backend":            type(self.store).__name__,
            "use_embeddings":     self.use_embeddings,
            "embedding_model":    self.embedding_model if self.use_embeddings else None,
            "embeddings_package": embeddings_available(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_payload_file(path: str | Path) -> list[dict[str, Any]]:
        """Load a JSON file containing a list of fork record dicts."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def export_json(self, path: str | Path) -> None:
        """Export all records to a JSON file."""
        records = self.load(include_expired=True)
        payload = [r.to_dict() for r in records]
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def import_json(self, path: str | Path) -> int:
        """Import records from a JSON file. Returns count of imported records."""
        payload = self.load_payload_file(path)
        records = self.add_records_from_payload(payload)
        return len(records)
