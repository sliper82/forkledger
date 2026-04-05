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
from .integrations.langchain import ForkLedgerMemory, langchain_available
from .integrations.autogen import (
    ForkLedgerHook,
    ForkLedgerGroupChatManager,
    autogen_available,
)

__version__ = "0.4.1"

__all__ = [
    # Core
    "Branch",
    "ForkRecord",
    "ForkLedgerEngine",
    "JsonForkStore",
    "SqliteForkStore",
    "OutcomeEstimate",
    # Retrieval
    "embeddings_available",
    "faiss_available",
    "ScoringWeights",
    "FaissIndex",
    # MCP
    "mcp_available",
    # Counterfactual
    "compute_regret",
    "fill_regret",
    "confidence_decay_factor",
    "accumulate_regret",
    "branch_win_rate",
    "normalized_regret",
    # Integrations
    "ForkLedgerMemory",
    "langchain_available",
    "ForkLedgerHook",
    "ForkLedgerGroupChatManager",
    "autogen_available",
    # Version
    "__version__",
]
