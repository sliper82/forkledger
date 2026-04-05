"""
ForkLedger × LangChain Integration
====================================
Official LangChain BaseMemory adapter.

Install:
    pip install forkledger[langchain]

Usage:
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationChain
    from forkledger.integrations.langchain import ForkLedgerMemory

    memory = ForkLedgerMemory(
        store_path="decisions.db",
        backend="sqlite",
        agent_id="my-agent",
    )

    chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

    # The chain automatically records decisions and retrieves
    # recommendations from past similar situations.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..engine import ForkLedgerEngine
from ..models import Branch, ForkRecord, OutcomeEstimate

import importlib.util as _ilu
_LANGCHAIN_AVAILABLE = _ilu.find_spec("langchain") is not None
# BaseMemory was removed in LangChain 0.2+ — use plain object as base
try:
    from langchain.memory.base import BaseMemory as _BaseMemory  # type: ignore
except ImportError:
    _BaseMemory = object


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required: pip install forkledger[langchain]\n"
            "  or: pip install langchain langchain-core"
        )


# ---------------------------------------------------------------------------
# ForkLedgerMemory — drop-in BaseMemory replacement
# ---------------------------------------------------------------------------

if _LANGCHAIN_AVAILABLE:
    class ForkLedgerMemory:
        """LangChain memory that stores decisions, alternatives, and regret.

        Unlike standard ConversationBufferMemory (which only stores messages),
        ForkLedgerMemory stores the *decision structure* of each interaction:
        - What state the agent was in
        - What options were available
        - What was chosen
        - What the outcome was
        - What it cost relative to alternatives

        This allows the agent to learn over time: similar future situations
        will receive recommendations based on what worked in the past.

        Parameters
        ----------
        store_path : str
            Path to the ForkLedger store file.
        backend : str
            "sqlite" (recommended) or "json".
        agent_id : str
            Unique identifier for this agent — used as namespace.
        memory_key : str
            Key injected into the chain's input dict. Default: "decision_context"
        top_k : int
            Number of historical forks to use for recommendations. Default: 3
        domain : str
            Scoring domain preset. One of: default, trading, research, code.
        auto_record : bool
            If True, automatically records each chain call as a decision.
        """

        def __init__(
            self,
            store_path: str = ".forkledger/langchain.db",
            backend: str = "sqlite",
            agent_id: str = "langchain-agent",
            memory_key: str = "decision_context",
            top_k: int = 3,
            domain: str = "default",
            auto_record: bool = True,
        ) -> None:
            self.store_path = store_path
            self.backend = backend
            self.agent_id = agent_id
            self.memory_key = memory_key
            self.top_k = top_k
            self.domain = domain
            self.auto_record = auto_record
            self._engine: ForkLedgerEngine | None = None

        def _get_engine(self) -> ForkLedgerEngine:
            if self._engine is None:
                self._engine = ForkLedgerEngine(
                    store_path=self.store_path,
                    backend=self.backend,  # type: ignore
                    confidence_half_life_days=60.0,
                )
            return self._engine

        @property
        def memory_variables(self) -> list[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
            """Load relevant decision context for the current input.

            Extracts the current state from inputs and returns recommendations
            from similar past decisions.
            """
            engine = self._get_engine()

            # Build state from inputs
            current_state = self._inputs_to_state(inputs)

            # Get recommendations from past decisions
            recs = engine.recommend(
                current_state=current_state,
                top_k=self.top_k,
                domain=self.domain,
                namespace=self.agent_id,
            )

            # Get distilled policies
            policies = engine.policies(
                min_support=2,
                namespace=self.agent_id,
            )

            # Build context string for the LLM
            context_parts = []

            if recs:
                context_parts.append("## Past Decision Recommendations")
                for i, rec in enumerate(recs, 1):
                    evidence = rec["support"][:2]
                    evidence_ids = [e["fork_id"] for e in evidence]
                    context_parts.append(
                        f"{i}. Branch '{rec['branch']}' — "
                        f"confidence score: {rec['score']:.3f} "
                        f"(based on: {evidence_ids})"
                    )

            if policies:
                context_parts.append("\n## Learned Policies")
                for p in policies[:3]:
                    context_parts.append(
                        f"- State {p['state']}: "
                        f"best branch is '{p['recommended_branch']}' "
                        f"(win rate: {p.get('win_rate', {}).get(p['recommended_branch'], 'N/A')})"
                    )

            if not context_parts:
                context_parts.append(
                    "No historical decision data yet. "
                    "Record decisions with memory.record_decision() to build history."
                )

            return {self.memory_key: "\n".join(context_parts)}

        def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
            """Save the current interaction as a decision record if auto_record=True."""
            if not self.auto_record:
                return

            engine = self._get_engine()
            current_state = self._inputs_to_state(inputs)

            # Extract chosen action from output
            output_text = outputs.get("response", outputs.get("output", ""))
            chosen = self._extract_action(output_text) or "respond"

            record = ForkRecord(
                fork_id=f"lc-{uuid.uuid4().hex[:12]}",
                pre_state=current_state,
                trigger=str(inputs.get("input", inputs.get("human_input", "interaction"))),
                possible_branches=[Branch(name=chosen), Branch(name="skip"), Branch(name="clarify")],
                chosen_branch=chosen,
                realized_value=0.0,  # update later with memory.update_outcome()
                confidence=0.6,
                tags=["langchain", self.agent_id],
                namespace=self.agent_id,
                outcome_source="observed",
            )
            engine.add_record(record)

        def clear(self) -> None:
            """No-op — ForkLedger memory is persistent by design."""
            pass

        # -- Helper methods --------------------------------------------------

        def record_decision(
            self,
            fork_id: str,
            situation: dict[str, Any] | str,
            trigger: str,
            chosen: str,
            alternatives: list[str],
            outcome: float,
            estimated_alternatives: dict[str, float] | None = None,
            confidence: float = 0.8,
            tags: list[str] | None = None,
        ) -> ForkRecord:
            """Explicitly record a decision with full structure.

            Args:
                fork_id: Unique ID for this decision.
                situation: State dict or string describing the situation.
                trigger: Why this decision was important.
                chosen: The branch that was taken.
                alternatives: Other available branches.
                outcome: Numeric value of the outcome (higher = better).
                estimated_alternatives: Expected value of each alternative.
                confidence: Record trustworthiness (0-1).
                tags: Labels for filtering.

            Returns:
                The recorded ForkRecord with computed regret vector.
            """
            engine = self._get_engine()

            if isinstance(situation, str):
                state = {"situation": situation}
            else:
                state = situation

            all_branches = [chosen] + [a for a in alternatives if a != chosen]
            estimated = []
            if estimated_alternatives:
                for name, val in estimated_alternatives.items():
                    estimated.append(OutcomeEstimate(branch_name=name, estimated_value=val))

            record = ForkRecord(
                fork_id=fork_id,
                pre_state=state,
                trigger=trigger,
                possible_branches=[Branch(name=b) for b in all_branches],
                chosen_branch=chosen,
                realized_value=outcome,
                estimated_outcomes=estimated,
                confidence=confidence,
                tags=(tags or []) + ["langchain", self.agent_id],
                namespace=self.agent_id,
                outcome_source="observed",
            )
            return engine.add_record(record)

        def update_outcome(
            self,
            fork_id: str,
            realized_value: float,
            confidence: float | None = None,
        ) -> ForkRecord | None:
            """Update the outcome of a previously recorded decision.

            Call this after observing the actual result of a decision.
            """
            return self._get_engine().update_outcome(fork_id, realized_value, confidence)

        def get_recommendations(
            self,
            situation: dict[str, Any] | str,
            top_k: int | None = None,
        ) -> list[dict[str, Any]]:
            """Get branch recommendations for a given situation."""
            if isinstance(situation, str):
                state = {"situation": situation}
            else:
                state = situation
            return self._get_engine().recommend(
                current_state=state,
                top_k=top_k or self.top_k,
                domain=self.domain,
                namespace=self.agent_id,
            )

        def get_policies(self) -> list[dict[str, Any]]:
            """Get distilled decision policies from history."""
            return self._get_engine().policies(
                min_support=2, namespace=self.agent_id
            )

        def stats(self) -> dict[str, Any]:
            """Return memory store statistics."""
            return self._get_engine().stats()

        # -- Private helpers -------------------------------------------------

        @staticmethod
        def _inputs_to_state(inputs: dict[str, Any]) -> dict[str, Any]:
            """Convert LangChain inputs dict to a ForkLedger state dict."""
            state: dict[str, Any] = {}
            skip_keys = {"decision_context", "chat_history", "history"}
            for k, v in inputs.items():
                if k in skip_keys:
                    continue
                if isinstance(v, str) and len(v) < 200:
                    state[k] = v
                elif isinstance(v, (int, float, bool)):
                    state[k] = v
            return state or {"input": str(list(inputs.values())[0])[:100] if inputs else "unknown"}

        @staticmethod
        def _extract_action(text: str) -> str | None:
            """Try to extract a named action from LLM output."""
            import re
            action_patterns = [
                r"action:\s*(\w+)",
                r"I will\s+(\w+)",
                r"choosing to\s+(\w+)",
                r"decision:\s*(\w+)",
            ]
            for pattern in action_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).lower()
            return None

else:
    # Stub when LangChain not installed
    class ForkLedgerMemory:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()


def langchain_available() -> bool:
    """Returns True if LangChain is installed."""
    return _LANGCHAIN_AVAILABLE
