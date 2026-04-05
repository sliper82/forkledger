"""
ForkLedger policy distillation — upgraded with fuzzy clustering.

Two modes:
  - exact   : original exact fingerprint match (fast, strict)
  - fuzzy   : groups similar states using overlap threshold (realistic)

The fuzzy mode solves the core problem: real-world agent states are never
identical between sessions. A situation with {"task":"trade","signal":"bullish","vol":"high"}
and {"task":"trade","signal":"bullish","vol":"very_high"} should be grouped.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import ForkRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Jaccard-style overlap between two state dicts."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    matches = sum(1 for k in all_keys if a.get(k) == b.get(k))
    return matches / len(all_keys)


def _find_or_create_cluster(
    state: dict[str, Any],
    clusters: list[dict[str, Any]],
    threshold: float,
) -> int:
    """Return index of best matching cluster, or -1 if none above threshold."""
    best_idx = -1
    best_sim = -1.0
    for i, centroid in enumerate(clusters):
        sim = _state_similarity(state, centroid)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    if best_sim >= threshold:
        return best_idx
    return -1


def _update_centroid(centroid: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Update cluster centroid: keep keys that match, drop those that diverge."""
    return {k: v for k, v in centroid.items() if state.get(k) == v}


# ---------------------------------------------------------------------------
# Exact policy distillation (original, preserved)
# ---------------------------------------------------------------------------

def distill_policies_exact(
    records: list[ForkRecord],
    min_support: int = 2,
) -> list[dict[str, Any]]:
    """Group records by exact state fingerprint."""
    grouped: dict[tuple, list[ForkRecord]] = defaultdict(list)
    for record in records:
        fingerprint = tuple(sorted(record.pre_state.items()))
        grouped[fingerprint].append(record)

    return _build_policies(
        [(dict(fp), group) for fp, group in grouped.items()],
        min_support=min_support,
    )


# ---------------------------------------------------------------------------
# Fuzzy policy distillation (new, recommended)
# ---------------------------------------------------------------------------

def distill_policies_fuzzy(
    records: list[ForkRecord],
    min_support: int = 2,
    similarity_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Group records by state similarity using greedy clustering.

    Records whose states overlap >= similarity_threshold are grouped together.
    This handles the real-world case where agent states are never perfectly identical.

    Args:
        records: ForkRecord list.
        min_support: Minimum records per cluster to emit a policy.
        similarity_threshold: Overlap ratio to assign to an existing cluster (0–1).
            0.5 = half the keys must match
            0.7 = most keys must match (default)
            1.0 = exact match only (same as exact mode)
    """
    centroids: list[dict[str, Any]] = []
    clusters: list[list[ForkRecord]] = []

    for record in records:
        idx = _find_or_create_cluster(record.pre_state, centroids, similarity_threshold)
        if idx == -1:
            centroids.append(dict(record.pre_state))
            clusters.append([record])
        else:
            clusters[idx].append(record)
            centroids[idx] = _update_centroid(centroids[idx], record.pre_state)

    return _build_policies(
        list(zip(centroids, clusters)),
        min_support=min_support,
    )


# ---------------------------------------------------------------------------
# Shared policy builder
# ---------------------------------------------------------------------------

def _build_policies(
    groups: list[tuple[dict[str, Any], list[ForkRecord]]],
    min_support: int,
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []

    for state_repr, group in groups:
        if len(group) < min_support:
            continue

        branch_stats: dict[str, list[float]] = defaultdict(list)
        branch_wins: dict[str, int] = defaultdict(int)
        branch_total: dict[str, int] = defaultdict(int)

        for record in group:
            for branch_name, regret in record.regret_vector.items():
                branch_stats[branch_name].append(regret)
                branch_total[branch_name] += 1
                if regret == 0.0:
                    branch_wins[branch_name] += 1

        if not branch_stats:
            continue

        avg_regret = {
            b: round(sum(v) / len(v), 6)
            for b, v in branch_stats.items()
        }
        win_rate = {
            b: round(branch_wins[b] / branch_total[b], 4)
            for b in branch_stats
        }
        best_branch = min(avg_regret, key=avg_regret.__getitem__)

        # Confidence: higher support + consistent winner = higher confidence
        chosen_counts: dict[str, int] = defaultdict(int)
        for r in group:
            chosen_counts[r.chosen_branch] += 1
        consistency = max(chosen_counts.values()) / len(group)

        policies.append({
            "state":              state_repr,
            "recommended_branch": best_branch,
            "average_regret":     {k: round(v, 6) for k, v in avg_regret.items()},
            "win_rate":           win_rate,
            "support":            len(group),
            "policy_confidence":  round(consistency * min(len(group) / 5, 1.0), 4),
        })

    policies.sort(key=lambda p: (p["support"], p["policy_confidence"]), reverse=True)
    return policies


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def distill_policies(
    records: list[ForkRecord],
    min_support: int = 2,
    fuzzy: bool = True,
    similarity_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Distill low-regret policies from repeated decision states.

    Args:
        records: All ForkRecord objects.
        min_support: Minimum records required to emit a policy.
        fuzzy: Use fuzzy clustering (recommended). Set False for exact matching.
        similarity_threshold: State overlap threshold for fuzzy mode (0.6 default).
    """
    if fuzzy:
        return distill_policies_fuzzy(records, min_support, similarity_threshold)
    return distill_policies_exact(records, min_support)
