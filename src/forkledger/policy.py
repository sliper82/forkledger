from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import ForkRecord


def distill_policies(records: list[ForkRecord], min_support: int = 2) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[str, Any], ...], list[ForkRecord]] = defaultdict(list)
    for record in records:
        fingerprint = tuple(sorted(record.pre_state.items()))
        grouped[fingerprint].append(record)

    policies: list[dict[str, Any]] = []
    for fingerprint, group in grouped.items():
        if len(group) < min_support:
            continue

        branch_stats: dict[str, list[float]] = defaultdict(list)
        for record in group:
            for branch_name, regret in record.regret_vector.items():
                branch_stats[branch_name].append(regret)

        if not branch_stats:
            continue

        avg_regret = {
            branch: sum(values) / len(values)
            for branch, values in branch_stats.items()
        }
        best_branch = min(avg_regret, key=avg_regret.get)
        policies.append(
            {
                "state": dict(fingerprint),
                "recommended_branch": best_branch,
                "average_regret": {k: round(v, 6) for k, v in avg_regret.items()},
                "support": len(group),
            }
        )

    policies.sort(key=lambda item: item["support"], reverse=True)
    return policies
