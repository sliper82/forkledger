from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import ForkRecord


def _to_dt_or_none(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _state_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    matches = sum(1 for key in keys if a.get(key) == b.get(key))
    return matches / max(len(keys), 1)


def _constraint_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _state_overlap(a, b)


def _recency_score(created_at: str) -> float:
    created = _to_dt_or_none(created_at)
    if created is None:
        return 0.5
    age_days = max((datetime.now(timezone.utc) - created).days, 0)
    return 1.0 / (1.0 + (age_days / 30.0))


def _salience(record: ForkRecord) -> float:
    if not record.regret_vector:
        return 0.0
    return max(record.regret_vector.values(), default=0.0)


def score_record(record: ForkRecord, current_state: dict[str, Any], constraints: dict[str, Any] | None = None) -> float:
    constraints = constraints or {}
    state_similarity = _state_overlap(record.pre_state, current_state)
    constraint_similarity = _constraint_overlap(record.constraints, constraints)
    recency = _recency_score(record.created_at)
    salience = _salience(record)
    confidence = max(min(record.confidence, 1.0), 0.0)

    score = (
        0.45 * state_similarity
        + 0.20 * constraint_similarity
        + 0.15 * recency
        + 0.10 * confidence
        + 0.10 * min(salience / 10.0, 1.0)
    )
    return round(score, 6)


def rank_records(records: list[ForkRecord], current_state: dict[str, Any], constraints: dict[str, Any] | None = None) -> list[tuple[ForkRecord, float]]:
    ranked = [(record, score_record(record, current_state, constraints)) for record in records]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def recommend_branches(records: list[ForkRecord], current_state: dict[str, Any], constraints: dict[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    ranked = rank_records(records, current_state, constraints)[:top_k]
    aggregate: Counter[str] = Counter()
    support: dict[str, list[dict[str, Any]]] = {}

    for record, score in ranked:
        for branch in record.possible_branches:
            regret = record.regret_vector.get(branch.name, 0.0)
            branch_score = max(0.0, score * (1.0 / (1.0 + regret)))
            aggregate[branch.name] += branch_score
            support.setdefault(branch.name, []).append(
                {
                    "fork_id": record.fork_id,
                    "historical_choice": record.chosen_branch,
                    "historical_regret": regret,
                    "match_score": score,
                }
            )

    recommendations = [
        {
            "branch": branch,
            "score": round(score, 6),
            "support": support.get(branch, []),
        }
        for branch, score in aggregate.most_common()
    ]
    return recommendations
