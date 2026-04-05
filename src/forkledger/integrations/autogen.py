"""
ForkLedger × AutoGen Integration
===================================
Decision memory for AutoGen agents and GroupChat workflows.

Install:
    pip install forkledger[autogen]

Usage — single agent:
    from autogen import AssistantAgent, UserProxyAgent
    from forkledger.integrations.autogen import ForkLedgerHook

    hook = ForkLedgerHook(store_path="decisions.db", agent_id="assistant")

    assistant = AssistantAgent("assistant", llm_config=llm_config)
    hook.attach(assistant)   # injects decision context into every message

Usage — GroupChat:
    from forkledger.integrations.autogen import ForkLedgerGroupChatManager

    manager = ForkLedgerGroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        store_path="decisions.db",
    )
"""

from __future__ import annotations

import uuid
from typing import Any

from ..engine import ForkLedgerEngine
from ..models import Branch, ForkRecord, OutcomeEstimate

import importlib.util as _ilu
_AUTOGEN_AVAILABLE = _ilu.find_spec("autogen") is not None


def _require_autogen() -> None:
    if not _AUTOGEN_AVAILABLE:
        raise ImportError(
            "AutoGen is required: pip install forkledger[autogen]\n"
            "  or: pip install pyautogen"
        )


# ---------------------------------------------------------------------------
# ForkLedgerHook — attaches to any AutoGen ConversableAgent
# ---------------------------------------------------------------------------

class ForkLedgerHook:
    """Hooks into an AutoGen agent to provide decision memory.

    Injects past decision context into each message the agent processes,
    and records new decisions as they happen.

    Parameters
    ----------
    store_path : str
        Path to the ForkLedger store.
    backend : str
        "sqlite" or "json".
    agent_id : str
        Namespace for this agent's decisions.
    top_k : int
        Historical forks to surface per query.
    domain : str
        Scoring domain: default, trading, research, code.
    inject_context : bool
        If True, prepends decision recommendations to each message.
    """

    def __init__(
        self,
        store_path: str = ".forkledger/autogen.db",
        backend: str = "sqlite",
        agent_id: str = "autogen-agent",
        top_k: int = 3,
        domain: str = "default",
        inject_context: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.top_k = top_k
        self.domain = domain
        self.inject_context = inject_context
        self._engine = ForkLedgerEngine(
            store_path=store_path,
            backend=backend,  # type: ignore
        )
        self._last_state: dict[str, Any] = {}

    def attach(self, agent: Any) -> None:
        """Attach this hook to an AutoGen ConversableAgent.

        Registers process_message_before_send and process_last_received_message hooks.
        """
        _require_autogen()
        if hasattr(agent, "register_hook"):
            agent.register_hook(
                "process_message_before_send",
                self._before_send,
            )
            agent.register_hook(
                "process_last_received_message",
                self._after_receive,
            )
        else:
            raise TypeError(
                f"Agent {type(agent)} does not support register_hook(). "
                "Requires AutoGen ConversableAgent or subclass."
            )

    def _before_send(self, message: dict[str, Any] | str, **kwargs: Any) -> dict[str, Any] | str:
        """Inject decision context before agent sends a message."""
        if not self.inject_context:
            return message

        context = self._build_context(self._last_state)
        if not context:
            return message

        prefix = f"[ForkLedger Decision Context]\n{context}\n\n"

        if isinstance(message, str):
            return prefix + message
        elif isinstance(message, dict) and "content" in message:
            content = message.get("content", "")
            if isinstance(content, str):
                message = dict(message)
                message["content"] = prefix + content
        return message

    def _after_receive(self, message: dict[str, Any] | str, **kwargs: Any) -> dict[str, Any] | str:
        """Extract state from received message for next context lookup."""
        if isinstance(message, str):
            self._last_state = {"message": message[:200]}
        elif isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                self._last_state = {"message": content[:200]}
            self._last_state.update({
                k: v for k, v in message.items()
                if k not in {"content", "role"} and isinstance(v, (str, int, float, bool))
            })
        return message

    def _build_context(self, state: dict[str, Any]) -> str:
        """Build decision context string from current state."""
        if not state:
            return ""
        recs = self._engine.recommend(
            current_state=state,
            top_k=self.top_k,
            domain=self.domain,
            namespace=self.agent_id,
        )
        if not recs:
            return ""
        lines = ["Recommended branches based on past decisions:"]
        for i, r in enumerate(recs, 1):
            lines.append(f"  {i}. '{r['branch']}' (score={r['score']:.3f})")
        return "\n".join(lines)

    # -- Public API ----------------------------------------------------------

    def record(
        self,
        situation: dict[str, Any] | str,
        trigger: str,
        chosen: str,
        alternatives: list[str],
        outcome: float,
        estimated: dict[str, float] | None = None,
        confidence: float = 0.8,
        tags: list[str] | None = None,
    ) -> ForkRecord:
        """Record a decision made by this agent."""
        if isinstance(situation, str):
            state = {"situation": situation}
        else:
            state = situation

        all_branches = [chosen] + [a for a in alternatives if a != chosen]
        est_list = [
            OutcomeEstimate(branch_name=n, estimated_value=v)
            for n, v in (estimated or {}).items()
        ]

        record = ForkRecord(
            fork_id=f"ag-{uuid.uuid4().hex[:12]}",
            pre_state=state,
            trigger=trigger,
            possible_branches=[Branch(name=b) for b in all_branches],
            chosen_branch=chosen,
            realized_value=outcome,
            estimated_outcomes=est_list,
            confidence=confidence,
            tags=(tags or []) + ["autogen", self.agent_id],
            namespace=self.agent_id,
        )
        return self._engine.add_record(record)

    def update(self, fork_id: str, outcome: float, confidence: float | None = None) -> ForkRecord | None:
        """Update outcome for a previously recorded decision."""
        return self._engine.update_outcome(fork_id, outcome, confidence)

    def recommend(self, situation: dict[str, Any] | str) -> list[dict[str, Any]]:
        """Get branch recommendations for a situation."""
        state = {"situation": situation} if isinstance(situation, str) else situation
        return self._engine.recommend(
            current_state=state, top_k=self.top_k,
            domain=self.domain, namespace=self.agent_id,
        )

    def policies(self) -> list[dict[str, Any]]:
        """Get distilled decision policies."""
        return self._engine.policies(min_support=2, namespace=self.agent_id)

    def stats(self) -> dict[str, Any]:
        return self._engine.stats()


# ---------------------------------------------------------------------------
# ForkLedgerGroupChatManager — GroupChat with shared decision memory
# ---------------------------------------------------------------------------

class ForkLedgerGroupChatManager:
    """GroupChat manager wrapper that gives all agents shared decision memory.

    Every agent in the group sees the same decision history and recommendations.
    Decisions recorded by any agent are visible to all others.

    Parameters
    ----------
    groupchat : autogen.GroupChat
    llm_config : dict
    store_path : str
    backend : str
    session_id : str
        Shared namespace for all agents in this group.
    top_k : int
    domain : str
    """

    def __init__(
        self,
        groupchat: Any,
        llm_config: dict[str, Any],
        store_path: str = ".forkledger/groupchat.db",
        backend: str = "sqlite",
        session_id: str | None = None,
        top_k: int = 3,
        domain: str = "default",
    ) -> None:
        _require_autogen()
        self.groupchat = groupchat
        self.llm_config = llm_config
        self.session_id = session_id or f"groupchat-{uuid.uuid4().hex[:8]}"
        self.top_k = top_k
        self.domain = domain

        self._engine = ForkLedgerEngine(
            store_path=store_path,
            backend=backend,  # type: ignore
        )

        # Attach hooks to all agents in the group
        self._hooks: list[ForkLedgerHook] = []
        for agent in groupchat.agents:
            hook = ForkLedgerHook(
                store_path=store_path,
                backend=backend,
                agent_id=self.session_id,  # shared namespace
                top_k=top_k,
                domain=domain,
            )
            try:
                hook.attach(agent)
                self._hooks.append(hook)
            except (TypeError, AttributeError):
                pass  # agent doesn't support hooks

    def record_group_decision(
        self,
        situation: dict[str, Any] | str,
        trigger: str,
        chosen_agent: str,
        alternative_agents: list[str],
        outcome: float,
        confidence: float = 0.8,
    ) -> ForkRecord:
        """Record which agent/action was chosen in a group decision."""
        state = {"situation": situation} if isinstance(situation, str) else situation

        return self._engine.add_record(ForkRecord(
            fork_id=f"gc-{uuid.uuid4().hex[:12]}",
            pre_state=state,
            trigger=trigger,
            possible_branches=[
                Branch(name=b)
                for b in [chosen_agent] + alternative_agents
            ],
            chosen_branch=chosen_agent,
            realized_value=outcome,
            confidence=confidence,
            tags=["autogen", "groupchat", self.session_id],
            namespace=self.session_id,
        ))

    def recommend_agent(self, situation: dict[str, Any] | str) -> list[dict[str, Any]]:
        """Recommend which agent to route to based on past decisions."""
        state = {"situation": situation} if isinstance(situation, str) else situation
        return self._engine.recommend(
            current_state=state, top_k=self.top_k,
            domain=self.domain, namespace=self.session_id,
        )

    def shared_policies(self) -> list[dict[str, Any]]:
        """Policies distilled from all agents' decisions."""
        return self._engine.policies(min_support=2, namespace=self.session_id)


def autogen_available() -> bool:
    """Returns True if AutoGen is installed."""
    return _AUTOGEN_AVAILABLE
