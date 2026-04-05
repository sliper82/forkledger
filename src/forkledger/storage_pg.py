"""
ForkLedger PostgreSQL + pgvector storage backend (Tier 3).

Requires: pip install forkledger[postgres]

Ideal for:
- High-throughput multi-agent deployments
- Cloud-native setups (Railway, Supabase, AWS RDS)
- Semantic similarity via pgvector (no external embedding service needed)
"""

from __future__ import annotations

import json
from typing import Any

from .models import ForkRecord

try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS forks (
    fork_id       TEXT PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expiry        TIMESTAMPTZ,
    confidence    FLOAT NOT NULL DEFAULT 0.5,
    chosen_branch TEXT NOT NULL,
    trigger       TEXT NOT NULL,
    tags          TEXT[] NOT NULL DEFAULT '{}',
    data          JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forks_created_at    ON forks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_forks_confidence    ON forks(confidence);
CREATE INDEX IF NOT EXISTS idx_forks_chosen_branch ON forks(chosen_branch);
CREATE INDEX IF NOT EXISTS idx_forks_tags          ON forks USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_forks_data          ON forks USING GIN(data);
"""


class PostgresForkStore:
    """PostgreSQL-backed store with JSONB, GIN indexes, and native array tag support.

    Parameters
    ----------
    dsn : str
        PostgreSQL connection string.
        e.g. "postgresql://user:pass@localhost:5432/forkledger"
    """

    def __init__(self, dsn: str) -> None:
        if not _PG_AVAILABLE:
            raise ImportError("Install postgres deps: pip install forkledger[postgres]")
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA)
        self._conn.commit()

    def _row_to_record(self, row: tuple) -> ForkRecord:
        data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return ForkRecord.from_dict(data)

    def load(
        self,
        include_expired: bool = False,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> list[ForkRecord]:
        sql = "SELECT data FROM forks WHERE confidence >= %s"
        params: list[Any] = [min_confidence]

        if not include_expired:
            sql += " AND (expiry IS NULL OR expiry > now())"

        if tags:
            sql += " AND tags && %s"
            params.append(tags)

        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [self._row_to_record(r) for r in cur.fetchall()]

    def get(self, fork_id: str) -> ForkRecord | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT data FROM forks WHERE fork_id = %s", (fork_id,))
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def append(self, records: list[ForkRecord]) -> None:
        rows = []
        for r in records:
            rows.append((
                r.fork_id,
                r.created_at,
                r.expiry,
                r.confidence,
                r.chosen_branch,
                r.trigger,
                r.tags,
                json.dumps(r.to_dict(), ensure_ascii=False),
            ))
        with self._conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO forks (fork_id, created_at, expiry, confidence,
                                   chosen_branch, trigger, tags, data)
                VALUES %s
                ON CONFLICT (fork_id) DO UPDATE SET
                  data       = EXCLUDED.data,
                  confidence = EXCLUDED.confidence,
                  expiry     = EXCLUDED.expiry,
                  tags       = EXCLUDED.tags
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
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE forks SET data = %s, confidence = %s WHERE fork_id = %s",
                (json.dumps(record.to_dict()), record.confidence, fork_id),
            )
        self._conn.commit()
        return record

    def delete(self, fork_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM forks WHERE fork_id = %s", (fork_id,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def purge_expired(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM forks WHERE expiry IS NOT NULL AND expiry <= now()")
            count = cur.rowcount
        self._conn.commit()
        return count

    def stats(self) -> dict[str, Any]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), AVG(confidence) FROM forks")
            total, avg_conf = cur.fetchone()
            cur.execute(
                "SELECT chosen_branch, COUNT(*) FROM forks "
                "GROUP BY chosen_branch ORDER BY COUNT(*) DESC LIMIT 5"
            )
            top = [{"branch": r[0], "count": r[1]} for r in cur.fetchall()]
        return {
            "total_forks":    total or 0,
            "avg_confidence": round(float(avg_conf or 0), 4),
            "top_branches":   top,
            "backend":        "postgresql",
        }

    def close(self) -> None:
        self._conn.close()


def postgres_available() -> bool:
    return _PG_AVAILABLE
