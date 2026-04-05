"""ForkLedger — branch-based memory engine for AI systems."""

from .engine import ForkLedgerEngine
from .models import Branch, ForkRecord, OutcomeEstimate
from .retrieval import embeddings_available
from .storage import JsonForkStore, SqliteForkStore

__version__ = "0.2.0"

__all__ = [
    "Branch",
    "ForkRecord",
    "ForkLedgerEngine",
    "JsonForkStore",
    "SqliteForkStore",
    "OutcomeEstimate",
    "embeddings_available",
    "__version__",
]
