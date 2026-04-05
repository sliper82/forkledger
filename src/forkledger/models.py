"""ForkLedger data models — v0.4.0.

Changes:
- Added `updated_at` field for audit trail
- Added `outcome_source` for provenance tracking
- Added `namespace` for multi-agent isolation
- Backward-compatible: all new fields have defaults
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Branch:
    name: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutcomeEstimate:
    branch_name: str
    estimated_value: float
    confidence: float = 0.5
    rationale: str = ""


OutcomeSource = Literal["observed", "simulated", "estimated", "human"]


@dataclass(slots=True)
class ForkRecord:
    fork_id: str
    pre_state: dict[str, Any]
    trigger: str
    possible_branches: list[Branch]
    chosen_branch: str
    realized_value: float
    # optional / computed
    estimated_outcomes: list[OutcomeEstimate] = field(default_factory=list)
    regret_vector: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    expiry: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)   # NEW: tracks outcome updates
    tags: list[str] = field(default_factory=list)
    namespace: str = "default"                              # NEW: multi-agent isolation
    outcome_source: OutcomeSource = "observed"              # NEW: provenance of realized_value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def touch(self) -> None:
        """Update the updated_at timestamp to now."""
        self.updated_at = utc_now_iso()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForkRecord":
        return cls(
            fork_id=payload["fork_id"],
            pre_state=dict(payload.get("pre_state", {})),
            trigger=payload.get("trigger", ""),
            possible_branches=[
                Branch(**b) if not isinstance(b, Branch) else b
                for b in payload.get("possible_branches", [])
            ],
            chosen_branch=payload["chosen_branch"],
            realized_value=float(payload.get("realized_value", 0.0)),
            estimated_outcomes=[
                OutcomeEstimate(**e) if not isinstance(e, OutcomeEstimate) else e
                for e in payload.get("estimated_outcomes", [])
            ],
            regret_vector={k: float(v) for k, v in payload.get("regret_vector", {}).items()},
            constraints=dict(payload.get("constraints", {})),
            provenance=dict(payload.get("provenance", {})),
            confidence=float(payload.get("confidence", 0.5)),
            expiry=payload.get("expiry"),
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", payload.get("created_at", utc_now_iso())),
            tags=list(payload.get("tags", [])),
            namespace=payload.get("namespace", "default"),
            outcome_source=payload.get("outcome_source", "observed"),
        )
