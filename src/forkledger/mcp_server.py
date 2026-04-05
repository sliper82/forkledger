"""
ForkLedger MCP Server
=====================
Exposes ForkLedger as a Model Context Protocol server.

Any MCP-compatible client (Claude Desktop, Cursor, VS Code, AutoGen, etc.)
can connect and use ForkLedger as a decision memory layer — no code required.

Install:
    pip install forkledger[mcp]

Run (stdio transport — for Claude Desktop / Cursor):
    forkledger mcp

Run (HTTP transport — for remote agents):
    forkledger mcp --transport http --port 8001

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "forkledger": {
          "command": "forkledger",
          "args": ["mcp", "--store", "/path/to/store.db", "--backend", "sqlite"]
        }
      }
    }
"""

from __future__ import annotations

import json
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

from .engine import ForkLedgerEngine
from .models import Branch, ForkRecord, OutcomeEstimate


def create_mcp_server(
    store_path: str = ".forkledger/store.db",
    backend: str = "sqlite",
    use_embeddings: bool = False,
) -> Any:
    """Create and return a FastMCP server instance."""
    if not _MCP_AVAILABLE:
        raise ImportError(
            "MCP support requires: pip install forkledger[mcp]\n"
            "  or: pip install mcp fastmcp"
        )

    engine = ForkLedgerEngine(
        store_path=store_path,
        backend=backend,          # type: ignore[arg-type]
        use_embeddings=use_embeddings,
    )

    mcp = FastMCP(
        name="ForkLedger",
        instructions=(
            "ForkLedger is a branch-based decision memory engine. "
            "Use it to record decisions with their alternatives and outcomes, "
            "retrieve recommendations based on similar past situations, "
            "and learn which branches historically produce the lowest regret. "
            "Always record important decisions and update outcomes after observing results."
        ),
    )

    # ── Tool: record_decision ────────────────────────────────────────────────

    @mcp.tool()
    def record_decision(
        fork_id: str,
        situation: str,
        trigger: str,
        chosen: str,
        alternatives: list[str],
        outcome: float,
        estimated_alternatives: dict[str, float] | None = None,
        confidence: float = 0.7,
        tags: list[str] | None = None,
        expiry_days: int | None = None,
    ) -> str:
        """Record a decision that was made, its alternatives, and the observed outcome.

        Args:
            fork_id: Unique ID for this decision (e.g. 'trade-2024-001').
            situation: Description of the situation as a dict-like string or JSON.
            trigger: Why this decision was important / consequential.
            chosen: The branch/action that was actually taken.
            alternatives: List of other options that were available.
            outcome: Numeric value of the outcome (higher = better).
            estimated_alternatives: Dict mapping alternative names to estimated values.
            confidence: How confident you are in this record (0.0–1.0).
            tags: Labels for filtering (e.g. ['crypto', 'trade']).
            expiry_days: Auto-expire this record after N days (None = never).
        """
        try:
            state = json.loads(situation) if situation.strip().startswith("{") else {"situation": situation}
        except json.JSONDecodeError:
            state = {"situation": situation}

        all_branches = [chosen] + [a for a in alternatives if a != chosen]
        possible = [Branch(name=b) for b in all_branches]

        estimated = []
        if estimated_alternatives:
            for name, val in estimated_alternatives.items():
                estimated.append(OutcomeEstimate(branch_name=name, estimated_value=val))

        expiry = None
        if expiry_days is not None:
            from datetime import datetime, timezone, timedelta
            expiry = (datetime.now(timezone.utc) + timedelta(days=expiry_days)).isoformat()

        record = ForkRecord(
            fork_id=fork_id,
            pre_state=state,
            trigger=trigger,
            possible_branches=possible,
            chosen_branch=chosen,
            realized_value=outcome,
            estimated_outcomes=estimated,
            confidence=max(0.0, min(1.0, confidence)),
            tags=tags or [],
            expiry=expiry,
        )
        result = engine.add_record(record)
        regret = result.regret_vector.get(chosen, 0.0)
        return (
            f"✓ Recorded decision '{fork_id}': chose '{chosen}' → outcome={outcome}. "
            f"Regret for chosen path: {regret:.3f}. "
            f"Regret vector: {result.regret_vector}"
        )

    # ── Tool: update_outcome ─────────────────────────────────────────────────

    @mcp.tool()
    def update_outcome(
        fork_id: str,
        realized_value: float,
        confidence: float | None = None,
    ) -> str:
        """Update the realized value of a previously recorded decision and recompute regret.

        Call this after you observe the actual result of a decision you recorded earlier.

        Args:
            fork_id: The ID of the decision to update.
            realized_value: The actual observed outcome value.
            confidence: Updated confidence in the record (optional).
        """
        record = engine.update_outcome(fork_id, realized_value, confidence)
        if record is None:
            return f"✗ No decision found with id='{fork_id}'"
        return (
            f"✓ Updated '{fork_id}': realized_value={realized_value}. "
            f"New regret vector: {record.regret_vector}"
        )

    # ── Tool: get_recommendation ─────────────────────────────────────────────

    @mcp.tool()
    def get_recommendation(
        situation: str,
        constraints: str = "{}",
        top_k: int = 5,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> str:
        """Get branch recommendations based on past decisions in similar situations.

        Returns the branches that historically produced the lowest regret
        under situations similar to the one described.

        Args:
            situation: Current situation as JSON string or plain description.
            constraints: Active hard constraints as JSON string.
            top_k: Number of top historical forks to consider (default 5).
            tags: Filter source decisions by tags.
            min_confidence: Minimum confidence threshold for source records.
        """
        try:
            state = json.loads(situation) if situation.strip().startswith("{") else {"situation": situation}
        except json.JSONDecodeError:
            state = {"situation": situation}

        try:
            cons = json.loads(constraints)
        except json.JSONDecodeError:
            cons = {}

        recs = engine.recommend(
            current_state=state,
            constraints=cons,
            top_k=top_k,
            tags=tags,
            min_confidence=min_confidence,
        )

        if not recs:
            return "No relevant historical decisions found. Record some decisions first."

        lines = ["Branch recommendations (ranked by historical low-regret score):\n"]
        for i, r in enumerate(recs, 1):
            evidence_ids = [s["fork_id"] for s in r["support"][:3]]
            lines.append(
                f"  #{i} '{r['branch']}' — score={r['score']:.4f} "
                f"| evidence: {evidence_ids}"
            )
        return "\n".join(lines)

    # ── Tool: get_policies ───────────────────────────────────────────────────

    @mcp.tool()
    def get_policies(
        min_support: int = 2,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> str:
        """Get distilled decision policies from repeated similar situations.

        Returns patterns: under state X, branch Y has historically been best.

        Args:
            min_support: Minimum number of similar past decisions required.
            tags: Filter source decisions by tags.
            min_confidence: Minimum confidence threshold.
        """
        policies = engine.policies(
            min_support=min_support,
            tags=tags,
            min_confidence=min_confidence,
        )
        if not policies:
            return f"Not enough data yet (need {min_support}+ similar decisions per state pattern)."

        lines = ["Distilled decision policies:\n"]
        for p in policies:
            lines.append(
                f"  State: {p['state']}\n"
                f"  → Best branch: '{p['recommended_branch']}' "
                f"(seen {p['support']} times, avg regret: {p['average_regret']})\n"
            )
        return "\n".join(lines)

    # ── Tool: recall_decision ────────────────────────────────────────────────

    @mcp.tool()
    def recall_decision(fork_id: str) -> str:
        """Recall the full details of a specific past decision by its ID.

        Args:
            fork_id: The ID of the decision to retrieve.
        """
        record = engine.get(fork_id)
        if record is None:
            return f"No decision found with id='{fork_id}'"
        d = record.to_dict()
        return json.dumps(d, indent=2, ensure_ascii=False)

    # ── Tool: list_decisions ─────────────────────────────────────────────────

    @mcp.tool()
    def list_decisions(
        limit: int = 20,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> str:
        """List recent decisions stored in ForkLedger.

        Args:
            limit: Maximum number of records to return.
            tags: Filter by tags.
            min_confidence: Minimum confidence threshold.
        """
        records = engine.load(limit=limit, tags=tags, min_confidence=min_confidence)
        if not records:
            return "No decisions recorded yet."
        lines = [f"Recent decisions ({len(records)} shown):\n"]
        for r in records:
            lines.append(
                f"  [{r.fork_id}] chose='{r.chosen_branch}' "
                f"outcome={r.realized_value} conf={r.confidence} "
                f"tags={r.tags} created={r.created_at[:10]}"
            )
        return "\n".join(lines)

    # ── Tool: memory_stats ───────────────────────────────────────────────────

    @mcp.tool()
    def memory_stats() -> str:
        """Return statistics about the ForkLedger decision memory store."""
        s = engine.stats()
        return json.dumps(s, indent=2)

    # ── Tool: delete_decision ────────────────────────────────────────────────

    @mcp.tool()
    def delete_decision(fork_id: str) -> str:
        """Delete a specific decision from memory.

        Args:
            fork_id: The ID of the decision to delete.
        """
        ok = engine.delete(fork_id)
        return f"✓ Deleted '{fork_id}'" if ok else f"✗ Not found: '{fork_id}'"

    return mcp


def mcp_available() -> bool:
    return _MCP_AVAILABLE
