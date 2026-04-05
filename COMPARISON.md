# ForkLedger vs Other AI Memory Systems

## The core distinction

Every other AI memory system answers: **"What happened?"**

ForkLedger answers: **"When this situation appeared before — what options existed, which path was chosen, and what did that choice cost?"**

This is a fundamentally different question. And it requires a fundamentally different data structure.

---

## Side-by-side comparison

| Feature | ForkLedger | Mem0 | Hindsight | mcp-memory | LangMem |
|---|---|---|---|---|---|
| Stores decision alternatives | ✅ | ❌ | ❌ | ❌ | ❌ |
| Regret / opportunity cost | ✅ | ❌ | ❌ | ❌ | ❌ |
| Counterfactual reasoning | ✅ | ❌ | ❌ | ❌ | ❌ |
| Policy distillation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Branch win rates | ✅ | ❌ | ❌ | ❌ | ❌ |
| Outcome update after the fact | ✅ | ❌ | ❌ | ❌ | ❌ |
| Confidence decay over time | ✅ | ✅ | ❌ | ❌ | ✅ |
| MCP server | ✅ | ✅ | ✅ | ✅ | ✅ |
| Zero required dependencies | ✅ | ❌ | ❌ | ❌ | ❌ |
| SQLite (no infra required) | ✅ | ❌ | ❌ | ✅ | ❌ |
| REST API | ✅ | ✅ | ✅ | ❌ | ❌ |
| Semantic retrieval (optional) | ✅ | ✅ | ✅ | ❌ | ✅ |
| Requires LLM API key | ❌ | ✅ | ✅ | ❌ | ✅ |
| Open source (fully) | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## When to use ForkLedger

**Use ForkLedger when your agent:**
- Makes repeated decisions and needs to improve over time
- Has multiple available actions and needs to learn which work best
- Needs an auditable trail of *why* it chose what it chose
- Operates in a domain where past mistakes should reduce future confidence

**Examples:**
- Trading agents that learn from past positions
- Research agents that learn which sources to trust
- Code review agents that learn which patterns cause bugs
- Any agent doing A/B-style decision making over time

---

## When NOT to use ForkLedger

**Use a different system when you need:**
- Conversational memory / user preference recall → use **Mem0**
- Factual knowledge storage / RAG → use **LangMem** or **Hindsight**
- General-purpose "remember this fact" storage → use **mcp-memory**
- Real-time streaming context → use **Zep**

---

## Complementary use

ForkLedger works alongside other memory systems, not instead of them.

A typical production agent might use:
- **Mem0** for user preferences and conversation history
- **ForkLedger** for decision memory and policy learning
- **Hindsight** or **LangMem** for factual knowledge retrieval

```python
# Example: combined stack
from mem0 import Memory
from forkledger import ForkLedgerEngine

user_memory = Memory()          # what the user prefers
decision_memory = ForkLedgerEngine("decisions.db", backend="sqlite")  # what worked

# Before deciding:
prefs = user_memory.search("risk tolerance", user_id=user_id)
rec = decision_memory.recommend(current_state)

# After deciding:
decision_memory.update_outcome(fork_id, realized_value)
```
