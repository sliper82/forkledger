"""
ForkLedger counterfactual regret computation — upgraded.

Adds:
- Confidence decay: older records contribute less to regret-based decisions
- Iterative regret update: re-applies CFR-style accumulation across history
- Normalized regret: scaled to [0,1] for cross-record comparison
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .models import ForkRecord


# ---------------------------------------------------------------------------
# Core regret computation
# ---------------------------------------------------------------------------

def compute_regret(record: ForkRecord) -> dict[str, float]:
    """Compute regret vector for a single fork record.

    regret(branch) = best_value - branch_value
    Zero regret = best available choice. Lower is better.
    """
    estimates = {item.branch_name: item.estimated_value for item in record.estimated_outcomes}
    branch_values = dict(estimates)
    branch_values[record.chosen_branch] = record.realized_value

    if not branch_values:
        return {}

    best_value = max(branch_values.values())
    return {branch: round(best_value - value, 6) for branch, value in branch_values.items()}


def fill_regret(record: ForkRecord) -> ForkRecord:
    """Compute and attach regret vector to a record in-place."""
    record.regret_vector = compute_regret(record)
    return record


# ---------------------------------------------------------------------------
# Confidence decay
# ---------------------------------------------------------------------------

def confidence_decay_factor(
    created_at: str,
    half_life_days: float = 60.0,
) -> float:
    """Exponential decay: records lose weight as they age.

    half_life_days=60 means a 60-day-old record contributes 50% weight.
    Uses: weight = e^(-lambda * age_days), lambda = ln(2) / half_life

    Args:
        created_at: ISO timestamp of when the record was created.
        half_life_days: Days until weight drops to 50%.

    Returns:
        Float in (0, 1]. Recent records → 1.0. Old records → near 0.
    """
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = max((datetime.now(timezone.utc) - created).days, 0)
    except (ValueError, TypeError):
        return 1.0

    lam = math.log(2) / max(half_life_days, 1.0)
    return math.exp(-lam * age_days)


def weighted_regret(
    record: ForkRecord,
    half_life_days: float = 60.0,
) -> dict[str, float]:
    """Return regret vector weighted by confidence and age decay.

    Combines:
      - record.confidence (user-assigned trust)
      - exponential time decay

    Used by iterative regret accumulation.
    """
    decay = confidence_decay_factor(record.created_at, half_life_days)
    weight = record.confidence * decay
    return {
        branch: round(regret * weight, 6)
        for branch, regret in record.regret_vector.items()
    }


# ---------------------------------------------------------------------------
# Iterative CFR-style regret accumulation
# ---------------------------------------------------------------------------

def accumulate_regret(
    records: list[ForkRecord],
    half_life_days: float = 60.0,
) -> dict[str, float]:
    """Accumulate weighted regret across all records per branch.

    Implements a simplified CFR-style sum:
      R(branch) = Σ weighted_regret(record, branch) for all records

    Lower accumulated regret = historically better branch.

    Args:
        records: All relevant ForkRecord objects.
        half_life_days: Decay half-life for time weighting.

    Returns:
        Dict mapping branch name → accumulated weighted regret.
    """
    accumulated: dict[str, float] = {}
    for record in records:
        for branch, regret in weighted_regret(record, half_life_days).items():
            accumulated[branch] = accumulated.get(branch, 0.0) + regret
    return {b: round(v, 6) for b, v in accumulated.items()}


def normalized_regret(regret_vector: dict[str, float]) -> dict[str, float]:
    """Normalize regret to [0, 1] range for cross-record comparison.

    Useful when comparing decisions with very different outcome scales.
    """
    if not regret_vector:
        return {}
    max_regret = max(regret_vector.values())
    if max_regret == 0:
        return {b: 0.0 for b in regret_vector}
    return {b: round(v / max_regret, 6) for b, v in regret_vector.items()}


def branch_win_rate(records: list[ForkRecord]) -> dict[str, dict[str, Any]]:
    """Compute win rate per branch across all records.

    A branch 'wins' a record if its regret is 0.0 (i.e. it was the best choice).

    Returns:
        Dict mapping branch → {"wins": int, "appearances": int, "win_rate": float}
    """
    wins: dict[str, int] = {}
    appearances: dict[str, int] = {}

    for record in records:
        for branch, regret in record.regret_vector.items():
            appearances[branch] = appearances.get(branch, 0) + 1
            if regret == 0.0:
                wins[branch] = wins.get(branch, 0) + 1

    result: dict[str, dict[str, Any]] = {}
    for branch, count in appearances.items():
        w = wins.get(branch, 0)
        result[branch] = {
            "wins":        w,
            "appearances": count,
            "win_rate":    round(w / count, 4),
        }
    return result
