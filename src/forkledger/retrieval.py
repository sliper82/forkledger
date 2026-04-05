"""ForkLedger retrieval layer.

Scoring pipeline:
  1. state_overlap  — keyword key/value match (always available)
  2. embedding_sim  — cosine similarity via sentence-transformers (optional)
  3. constraint_overlap
  4. recency
  5. confidence
  6. regret_salience

Embedding support is opt-in. If sentence-transformers is not installed,
the engine falls back gracefully to keyword scoring only.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import ForkRecord

# ---------------------------------------------------------------------------
# Optional embedding support
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_encoder: Any = None  # lazy singleton


def _get_encoder(model: str = _DEFAULT_MODEL) -> Any:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(model)
    return _encoder


def _state_to_text(state: dict[str, Any]) -> str:
    """Convert a state dict to a flat string for embedding."""
    return " ".join(f"{k}={v}" for k, v in sorted(state.items()))


def _cosine(a: Any, b: Any) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


def embedding_similarity(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    model: str = _DEFAULT_MODEL,
) -> float:
    """Return cosine similarity between two states using sentence-transformers."""
    if not _EMBEDDINGS_AVAILABLE:
        return _state_overlap(state_a, state_b)   # graceful fallback
    enc = _get_encoder(model)
    text_a = _state_to_text(state_a)
    text_b = _state_to_text(state_b)
    vecs = enc.encode([text_a, text_b], normalize_embeddings=True)
    return float(_cosine(vecs[0], vecs[1]))


# ---------------------------------------------------------------------------
# Keyword scoring helpers
# ---------------------------------------------------------------------------

def _state_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    matches = sum(1 for k in keys if a.get(k) == b.get(k))
    return matches / max(len(keys), 1)


def _constraint_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _state_overlap(a, b)


def _recency_score(created_at: str) -> float:
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = max((datetime.now(timezone.utc) - created).days, 0)
        return 1.0 / (1.0 + (age_days / 30.0))
    except (ValueError, TypeError):
        return 0.5


def _salience(record: ForkRecord) -> float:
    return max(record.regret_vector.values(), default=0.0)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_record(
    record: ForkRecord,
    current_state: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    use_embeddings: bool = False,
    embedding_model: str = _DEFAULT_MODEL,
) -> float:
    """Score a single record against the current query state.

    Weights (sum to 1.0):
      0.45  state similarity   (embedding-based if use_embeddings=True)
      0.20  constraint match
      0.15  recency
      0.10  confidence
      0.10  regret salience (capped)
    """
    constraints = constraints or {}

    if use_embeddings and _EMBEDDINGS_AVAILABLE:
        state_sim = embedding_similarity(record.pre_state, current_state, model=embedding_model)
    else:
        state_sim = _state_overlap(record.pre_state, current_state)

    constraint_sim = _constraint_overlap(record.constraints, constraints)
    recency        = _recency_score(record.created_at)
    confidence     = max(min(record.confidence, 1.0), 0.0)
    salience       = min(_salience(record) / 10.0, 1.0)

    score = (
        0.45 * state_sim
        + 0.20 * constraint_sim
        + 0.15 * recency
        + 0.10 * confidence
        + 0.10 * salience
    )
    return round(score, 6)


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------

def rank_records(
    records: list[ForkRecord],
    current_state: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    use_embeddings: bool = False,
    embedding_model: str = _DEFAULT_MODEL,
) -> list[tuple[ForkRecord, float]]:
    ranked = [
        (r, score_record(r, current_state, constraints, use_embeddings, embedding_model))
        for r in records
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def recommend_branches(
    records: list[ForkRecord],
    current_state: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    top_k: int = 5,
    use_embeddings: bool = False,
    embedding_model: str = _DEFAULT_MODEL,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Return ranked branch recommendations with supporting evidence."""
    ranked = rank_records(records, current_state, constraints, use_embeddings, embedding_model)
    ranked = [(r, s) for r, s in ranked if s >= min_score][:top_k]

    aggregate: Counter[str] = Counter()
    support: dict[str, list[dict[str, Any]]] = {}

    for record, score in ranked:
        for branch in record.possible_branches:
            regret = record.regret_vector.get(branch.name, 0.0)
            branch_score = max(0.0, score * (1.0 / (1.0 + regret)))
            aggregate[branch.name] += branch_score
            support.setdefault(branch.name, []).append(
                {
                    "fork_id":           record.fork_id,
                    "historical_choice": record.chosen_branch,
                    "historical_regret": regret,
                    "match_score":       round(score, 6),
                }
            )

    return [
        {
            "branch":  branch,
            "score":   round(score, 6),
            "support": support.get(branch, []),
        }
        for branch, score in aggregate.most_common()
    ]


def embeddings_available() -> bool:
    """Returns True if sentence-transformers is installed."""
    return _EMBEDDINGS_AVAILABLE
