"""ForkLedger — branch-based decision memory for AI systems."""

from .engine import ForkLedgerEngine
from .models import Branch, ForkRecord, OutcomeEstimate
from .retrieval import embeddings_available, faiss_available, ScoringWeights, FaissIndex
from .storage import JsonForkStore, SqliteForkStore
from .mcp_server import mcp_available
from .counterfactual import (
    compute_regret,
    fill_regret,
    confidence_decay_factor,
    accumulate_regret,
    branch_win_rate,
    normalized_regret,
)

__version__ = "0.4.0"

__all__ = [
    "Branch",
    "ForkRecord",
    "ForkLedgerEngine",
    "JsonForkStore",
    "SqliteForkStore",
    "OutcomeEstimate",
    "embeddings_available",
    "faiss_available",
    "ScoringWeights",
    "FaissIndex",
    "mcp_available",
    "compute_regret",
    "fill_regret",
    "confidence_decay_factor",
    "accumulate_regret",
    "branch_win_rate",
    "normalized_regret",
    "__version__",
]
