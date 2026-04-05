"""
ForkLedger storage backends — v0.4.0

Changes:
- SqliteForkStore: FTS5 full-text search (hybrid keyword + semantic)
- SqliteForkStore: namespace isolation for multi-agent setups
- SqliteForkStore: updated_at column indexed
- SqliteForkStore: outcome_source column
- Both stores: touch() updated_at on outcome update
- Migration: auto-adds missing columns on open (backward compat)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ForkRecord


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(record: ForkRecord) -> bool:
    if not record.expiry:
        return False
    try:
        expiry_dt = datetime.fromisoformat(record.expiry.replace("Z", "+00:00"))
        return _now_utc() > expiry_dt
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# JSON store
# ---------------------------------------------------------------------------

class JsonForkStore:
    def __init__(self, path: str | Path = ".forkledger/store.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(
        self,
        include_expired: bool = False,
        namespace: str | None = None,
    ) -> list[ForkRecord]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = [ForkRecord.from_dict(item) for item in payload]
        if not include_expired:
            records = [r for r in records if not _is_expired(r)]
        if namespace:
            records = [r for r in records if r.namespace == namespace]
        return records

    def save(self, records: list[ForkRecord]) -> None:
        payload = [r.to_dict() for r in records]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, records: list[ForkRecord]) -> None:
        existing = {r.fork_id: r for r in self.load(include_expired=True)}
        for r in records:
            existing[r.fork_id] = r
        self.save(list(existing.values()))

    def update_outcome(
        self,
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
        outcome_source: str = "observed",
    ) -> ForkRecord | None:
        from .counterfactual import fill_regret
        records = {r.fork_id: r for r in self.load(include_expired=True)}
        if fork_id not in records:
            return None
        record = records[fork_id]
        record.realized_value = realized_value
        record.outcome_source = outcome_source   # type: ignore[attr-defined]
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
        record.touch()
        record = fill_regret(record)
        records[fork_id] = record
        self.save(list(records.values()))
        return record

    def delete(self, fork_id: str) -> bool:
        records = {r.fork_id: r for r in self.load(include_expired=True)}
        if fork_id not in records:
            return False
        del records[fork_id]
        self.save(list(records.values()))
        return True

    def purge_expired(self) -> int:
        all_records = self.load(include_expired=True)
        live = [r for r in all_records if not _is_expired(r)]
        removed = len(all_records) - len(live)
        self.save(live)
        return removed


# ---------------------------------------------------------------------------
# SQLite store — production backend
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forks (
    fork_id        TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL DEFAULT '',
    expiry         TEXT,
    confidence     REAL NOT NULL DEFAULT 0.5,
    chosen_branch  TEXT NOT NULL,
    trigger        TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT '[]',
    namespace      TEXT NOT NULL DEFAULT 'default',
    outcome_source TEXT NOT NULL DEFAULT 'observed',
    data           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forks_created_at    ON forks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_forks_updated_at    ON forks(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_forks_confidence    ON forks(confidence);
CREATE INDEX IF NOT EXISTS idx_forks_chosen_branch ON forks(chosen_branch);
CREATE INDEX IF NOT EXISTS idx_forks_trigger       ON forks(trigger);
CREATE INDEX IF NOT EXISTS idx_forks_namespace     ON forks(namespace);

-- FTS5 virtual table for full-text search (standalone, not content-backed)
CREATE VIRTUAL TABLE IF NOT EXISTS forks_fts USING fts5(
    fork_id,
    trigger,
    chosen_branch,
    tags,
    state_text,
    tokenize='porter ascii'
);
"""

# Columns added in v0.4.0 — auto-migrated on open
_MIGRATION_COLUMNS = [
    ("updated_at",     "TEXT NOT NULL DEFAULT ''"),
    ("namespace",      "TEXT NOT NULL DEFAULT 'default'"),
    ("outcome_source", "TEXT NOT NULL DEFAULT 'observed'"),
]


class SqliteForkStore:
    def __init__(self, path: str | Path = ".forkledger/store.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add new columns if upgrading from older schema."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(forks)").fetchall()
        }
        for col_name, col_def in _MIGRATION_COLUMNS:
            if col_name not in existing:
                self._conn.execute(
                    f"ALTER TABLE forks ADD COLUMN {col_name} {col_def}"
                )
        self._conn.commit()

    # -- helpers -----------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> ForkRecord:
        return ForkRecord.from_dict(json.loads(row["data"]))

    def _record_to_row(self, record: ForkRecord) -> dict[str, Any]:
        return {
            "fork_id":        record.fork_id,
            "created_at":     record.created_at,
            "updated_at":     record.updated_at,
            "expiry":         record.expiry,
            "confidence":     record.confidence,
            "chosen_branch":  record.chosen_branch,
            "trigger":        record.trigger,
            "tags":           json.dumps(record.tags),
            "namespace":      record.namespace,
            "outcome_source": record.outcome_source,
            "data":           json.dumps(record.to_dict(), ensure_ascii=False),
            # FTS fields
        }

    def _upsert_fts(self, record: ForkRecord) -> None:
        state_text = " ".join(f"{k}={v}" for k, v in sorted(record.pre_state.items()))
        self._conn.execute(
            "DELETE FROM forks_fts WHERE fork_id = ?", (record.fork_id,)
        )
        self._conn.execute(
            """INSERT INTO forks_fts(fork_id, trigger, chosen_branch, tags, state_text)
               VALUES (?, ?, ?, ?, ?)""",
            (record.fork_id, record.trigger, record.chosen_branch,
             " ".join(record.tags), state_text)
        )

    # -- read --------------------------------------------------------------

    def load(
        self,
        include_expired: bool = False,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
        namespace: str | None = None,
        full_text_query: str | None = None,
    ) -> list[ForkRecord]:
        """Load records with optional filters.

        Args:
            full_text_query: SQLite FTS5 query string for hybrid search.
                e.g. "breakout bullish" or "trade AND bullish"
        """
        # FTS5 hybrid search path
        if full_text_query:
            return self._fts_search(
                full_text_query, min_confidence, namespace, limit or 50
            )

        sql = "SELECT * FROM forks WHERE confidence >= ?"
        params: list[Any] = [min_confidence]

        if not include_expired:
            now_iso = _now_utc().isoformat()
            sql += " AND (expiry IS NULL OR expiry > ?)"
            params.append(now_iso)

        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)

        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, params).fetchall()
        records = [self._row_to_record(r) for r in rows]

        if tags:
            tag_set = set(tags)
            records = [r for r in records if tag_set.intersection(r.tags)]

        return records

    def _fts_search(
        self,
        query: str,
        min_confidence: float,
        namespace: str | None,
        limit: int,
    ) -> list[ForkRecord]:
        """Hybrid FTS5 full-text search combined with confidence filter."""
        try:
            # Get matching fork_ids from FTS table
            fts_sql = "SELECT fork_id FROM forks_fts WHERE forks_fts MATCH ? ORDER BY rank LIMIT ?"
            fts_rows = self._conn.execute(fts_sql, (query, limit * 2)).fetchall()
            if not fts_rows:
                return []
            fork_ids = [r[0] for r in fts_rows]
            placeholders = ",".join("?" * len(fork_ids))
            sql = f"SELECT * FROM forks WHERE fork_id IN ({placeholders}) AND confidence >= ?"
            params: list[Any] = fork_ids + [min_confidence]
            if namespace:
                sql += " AND namespace = ?"
                params.append(namespace)
            sql += f" LIMIT {int(limit)}"
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_record(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def _rebuild_fts(self) -> None:
        self._conn.execute("INSERT INTO forks_fts(forks_fts) VALUES('rebuild')")
        self._conn.commit()

    def get(self, fork_id: str) -> ForkRecord | None:
        row = self._conn.execute(
            "SELECT * FROM forks WHERE fork_id = ?", (fork_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    # -- write -------------------------------------------------------------

    def append(self, records: list[ForkRecord]) -> None:
        rows = [self._record_to_row(r) for r in records]
        self._conn.executemany(
            """INSERT INTO forks
               (fork_id, created_at, updated_at, expiry, confidence,
                chosen_branch, trigger, tags, namespace, outcome_source, data)
               VALUES (:fork_id, :created_at, :updated_at, :expiry, :confidence,
                       :chosen_branch, :trigger, :tags, :namespace, :outcome_source, :data)
               ON CONFLICT(fork_id) DO UPDATE SET
                 data           = excluded.data,
                 updated_at     = excluded.updated_at,
                 confidence     = excluded.confidence,
                 expiry         = excluded.expiry,
                 tags           = excluded.tags,
                 namespace      = excluded.namespace,
                 outcome_source = excluded.outcome_source
            """,
            rows,
        )
        for r in records:
            self._upsert_fts(r)
        self._conn.commit()

    def update_outcome(
        self,
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
        outcome_source: str = "observed",
    ) -> ForkRecord | None:
        from .counterfactual import fill_regret
        record = self.get(fork_id)
        if record is None:
            return None
        record.realized_value = realized_value
        record.outcome_source = outcome_source   # type: ignore[attr-defined]
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
        record.touch()
        record = fill_regret(record)
        row = self._record_to_row(record)
        self._conn.execute(
            """UPDATE forks SET
               data=?, confidence=?, updated_at=?, outcome_source=?
               WHERE fork_id=?""",
            (row["data"], row["confidence"], row["updated_at"],
             row["outcome_source"], fork_id),
        )
        self._upsert_fts(record)
        self._conn.commit()
        return record

    def delete(self, fork_id: str) -> bool:
        self._conn.execute("DELETE FROM forks_fts WHERE fork_id = ?", (fork_id,))
        cur = self._conn.execute("DELETE FROM forks WHERE fork_id = ?", (fork_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def purge_expired(self) -> int:
        now_iso = _now_utc().isoformat()
        expired = self._conn.execute(
            "SELECT fork_id FROM forks WHERE expiry IS NOT NULL AND expiry <= ?",
            (now_iso,)
        ).fetchall()
        for row in expired:
            self._conn.execute("DELETE FROM forks_fts WHERE fork_id = ?", (row["fork_id"],))
        cur = self._conn.execute(
            "DELETE FROM forks WHERE expiry IS NOT NULL AND expiry <= ?", (now_iso,)
        )
        self._conn.commit()
        return cur.rowcount

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM forks").fetchone()[0]
        expired_count = self._conn.execute(
            "SELECT COUNT(*) FROM forks WHERE expiry IS NOT NULL AND expiry <= ?",
            (_now_utc().isoformat(),),
        ).fetchone()[0]
        avg_conf = self._conn.execute("SELECT AVG(confidence) FROM forks").fetchone()[0]
        top_branches = self._conn.execute(
            "SELECT chosen_branch, COUNT(*) as cnt FROM forks "
            "GROUP BY chosen_branch ORDER BY cnt DESC LIMIT 5"
        ).fetchall()
        namespaces = self._conn.execute(
            "SELECT namespace, COUNT(*) as cnt FROM forks "
            "GROUP BY namespace ORDER BY cnt DESC"
        ).fetchall()
        return {
            "total_forks":    total,
            "expired_forks":  expired_count,
            "avg_confidence": round(avg_conf or 0.0, 4),
            "top_branches":   [{"branch": r["chosen_branch"], "count": r["cnt"]} for r in top_branches],
            "namespaces":     [{"namespace": r["namespace"], "count": r["cnt"]} for r in namespaces],
            "fts_enabled":    True,
            "wal_mode":       True,
        }

    def search_text(self, query: str, limit: int = 20) -> list[ForkRecord]:
        """Pure FTS5 full-text search across trigger, state, branch, tags."""
        return self._fts_search(query, 0.0, None, limit)

    def close(self) -> None:
        self._conn.close()
