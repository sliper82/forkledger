"""ForkLedger storage backends.

Provides:
- JsonForkStore  : original flat-file store (simple, zero-dependency)
- SqliteForkStore: SQLite-backed store with indexing, expiry, deduplication
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ForkRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# JSON store (original, kept for backward compat)
# ---------------------------------------------------------------------------

class JsonForkStore:
    """Flat JSON file store. Zero dependencies, works out of the box.
    Not recommended for >5 000 records."""

    def __init__(self, path: str | Path = ".forkledger/store.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self, include_expired: bool = False) -> list[ForkRecord]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = [ForkRecord.from_dict(item) for item in payload]
        if not include_expired:
            records = [r for r in records if not _is_expired(r)]
        return records

    def save(self, records: list[ForkRecord]) -> None:
        payload = [record.to_dict() for record in records]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, records: list[ForkRecord]) -> None:
        existing = {r.fork_id: r for r in self.load(include_expired=True)}
        for r in records:
            existing[r.fork_id] = r  # deduplicates by fork_id
        self.save(list(existing.values()))

    def update_outcome(self, fork_id: str, realized_value: float, confidence: float | None = None) -> ForkRecord | None:
        from .counterfactual import fill_regret
        records = {r.fork_id: r for r in self.load(include_expired=True)}
        if fork_id not in records:
            return None
        record = records[fork_id]
        record.realized_value = realized_value
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
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
# SQLite store (recommended for production)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forks (
    fork_id       TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    expiry        TEXT,
    confidence    REAL NOT NULL DEFAULT 0.5,
    chosen_branch TEXT NOT NULL,
    trigger       TEXT NOT NULL,
    tags          TEXT NOT NULL DEFAULT '[]',
    data          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forks_created_at    ON forks(created_at);
CREATE INDEX IF NOT EXISTS idx_forks_confidence    ON forks(confidence);
CREATE INDEX IF NOT EXISTS idx_forks_chosen_branch ON forks(chosen_branch);
CREATE INDEX IF NOT EXISTS idx_forks_trigger       ON forks(trigger);
"""


class SqliteForkStore:
    """SQLite-backed store.

    - Indexed columns for fast filtering
    - Expiry enforcement at query time
    - Atomic upserts (deduplication by fork_id)
    - Outcome update without full reload
    """

    def __init__(self, path: str | Path = ".forkledger/store.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> ForkRecord:
        return ForkRecord.from_dict(json.loads(row["data"]))

    def _record_to_row(self, record: ForkRecord) -> dict[str, Any]:
        return {
            "fork_id":       record.fork_id,
            "created_at":    record.created_at,
            "expiry":        record.expiry,
            "confidence":    record.confidence,
            "chosen_branch": record.chosen_branch,
            "trigger":       record.trigger,
            "tags":          json.dumps(record.tags),
            "data":          json.dumps(record.to_dict(), ensure_ascii=False),
        }

    def load(
        self,
        include_expired: bool = False,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> list[ForkRecord]:
        sql = "SELECT * FROM forks WHERE confidence >= ?"
        params: list[Any] = [min_confidence]

        if not include_expired:
            now_iso = _now_utc().isoformat()
            sql += " AND (expiry IS NULL OR expiry > ?)"
            params.append(now_iso)

        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, params).fetchall()
        records = [self._row_to_record(r) for r in rows]

        if tags:
            tag_set = set(tags)
            records = [r for r in records if tag_set.intersection(r.tags)]

        return records

    def get(self, fork_id: str) -> ForkRecord | None:
        row = self._conn.execute(
            "SELECT * FROM forks WHERE fork_id = ?", (fork_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def append(self, records: list[ForkRecord]) -> None:
        rows = [self._record_to_row(r) for r in records]
        self._conn.executemany(
            """INSERT INTO forks
               (fork_id, created_at, expiry, confidence, chosen_branch, trigger, tags, data)
               VALUES (:fork_id, :created_at, :expiry, :confidence, :chosen_branch, :trigger, :tags, :data)
               ON CONFLICT(fork_id) DO UPDATE SET
                 data       = excluded.data,
                 confidence = excluded.confidence,
                 expiry     = excluded.expiry,
                 tags       = excluded.tags
            """,
            rows,
        )
        self._conn.commit()

    def update_outcome(
        self,
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
    ) -> ForkRecord | None:
        from .counterfactual import fill_regret
        record = self.get(fork_id)
        if record is None:
            return None
        record.realized_value = realized_value
        if confidence is not None:
            record.confidence = max(0.0, min(1.0, confidence))
        record = fill_regret(record)
        row = self._record_to_row(record)
        self._conn.execute(
            "UPDATE forks SET data = ?, confidence = ? WHERE fork_id = ?",
            (row["data"], row["confidence"], fork_id),
        )
        self._conn.commit()
        return record

    def delete(self, fork_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM forks WHERE fork_id = ?", (fork_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def purge_expired(self) -> int:
        now_iso = _now_utc().isoformat()
        cur = self._conn.execute(
            "DELETE FROM forks WHERE expiry IS NOT NULL AND expiry <= ?", (now_iso,)
        )
        self._conn.commit()
        return cur.rowcount

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
        return {
            "total_forks":    total,
            "expired_forks":  expired_count,
            "avg_confidence": round(avg_conf or 0.0, 4),
            "top_branches":   [{"branch": r["chosen_branch"], "count": r["cnt"]} for r in top_branches],
        }

    def close(self) -> None:
        self._conn.close()
