<p align="center">
  <img src=".github/assets/logo.png" width="700" alt="ForkLedger" />
</p>

<p align="center">
  <a href="https://github.com/sliper82/forkledger/actions"><img src="https://github.com/sliper82/forkledger/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/forkledger/"><img src="https://img.shields.io/pypi/v/forkledger?color=5a32ff" alt="PyPI" /></a>
  <a href="https://pypi.org/project/forkledger/"><img src="https://img.shields.io/pypi/pyversions/forkledger" alt="Python" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License" /></a>
</p>

<p align="center">
  <b>The memory layer your AI agents actually need.</b><br/>
  Not what was said — but what was chosen, what it cost, and what to do differently next time.
</p>

---

## The problem with AI memory today

Every major memory system stores **what happened**:
messages, chunks, embeddings, graph edges.

None of them store **why a decision was made** — or what it cost.

That means every time your agent faces the same situation, it starts from zero. It can't learn from its own mistakes. It has no way to say: *"Last time I did this, it cost me. Don't do it again."*

**ForkLedger fixes this.**

---

## What ForkLedger stores

A ForkLedger **fork** is the atomic unit of decision memory:

```
situation before the choice
    ↓
[ branch A ] [ branch B ✓ chosen ] [ branch C ]
                    ↓
              observed outcome
                    ↓
         regret vector: what each path cost
```

Over time, ForkLedger learns which branches produce the lowest regret under similar conditions — and surfaces that knowledge when it matters.

---

## Install

```bash
# Zero dependencies core
pip install forkledger

# With REST API
pip install forkledger[api]

# With semantic similarity (sentence-transformers)
pip install forkledger[embeddings]

# Everything
pip install forkledger[all]
```

---

## 60-second demo

```python
from forkledger import ForkLedgerEngine, ForkRecord, Branch, OutcomeEstimate

engine = ForkLedgerEngine(".forkledger/store.db", backend="sqlite")

# Record a decision
engine.add_record(ForkRecord(
    fork_id    = "trade-2024-001",
    pre_state  = {"market": "BTC", "signal": "breakout", "volume": "high"},
    trigger    = "RSI crossed 70 with volume confirmation",
    possible_branches = [
        Branch(name="enter"),
        Branch(name="wait"),
        Branch(name="short"),
    ],
    chosen_branch    = "enter",
    realized_value   = 2.4,
    estimated_outcomes = [
        OutcomeEstimate(branch_name="wait",  estimated_value=0.0),
        OutcomeEstimate(branch_name="short", estimated_value=-1.5),
    ],
    confidence = 0.8,
    tags       = ["crypto", "momentum"],
))

# Later: update with actual outcome
engine.update_outcome("trade-2024-001", realized_value=1.9)

# Ask: given this new situation, what should I do?
recs = engine.recommend(
    current_state = {"market": "ETH", "signal": "breakout", "volume": "high"},
    top_k = 3,
)
# → [{"branch": "enter", "score": 0.82, "support": [...]}, ...]
```

---

## REST API

```bash
pip install forkledger[api]
forkledger serve --backend sqlite --store store.db
# → http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Store statistics |
| `GET` | `/forks` | List forks (filterable) |
| `POST` | `/forks` | Add a fork |
| `PATCH` | `/forks/{id}/outcome` | Update realized value |
| `DELETE` | `/forks/{id}` | Delete a fork |
| `POST` | `/recommend` | Get branch recommendations |
| `GET` | `/policies` | Distilled low-regret policies |

Full Swagger UI at `/docs`.

---

## CLI

```bash
# Record a decision
forkledger add decisions.json

# What should I do given this state?
forkledger recommend --state '{"signal": "breakout", "volume": "high"}' --top-k 3

# Update outcome after the fact
forkledger update-outcome trade-001 1.9 --confidence 0.85

# What patterns emerge from repeated states?
forkledger policies --min-support 3

# Start API server
forkledger serve --backend sqlite --port 8000

# Stats
forkledger stats
```

---

## Semantic similarity (optional)

```bash
pip install forkledger[embeddings]
```

```python
engine = ForkLedgerEngine(
    ".forkledger/store.db",
    backend       = "sqlite",
    use_embeddings = True,           # cosine similarity via sentence-transformers
    embedding_model = "all-MiniLM-L6-v2",
)
```

Without `sentence-transformers` installed, ForkLedger falls back silently to keyword overlap scoring. Zero breaking changes.

---

## Storage backends

| Backend | Best for | Deps |
|---------|----------|------|
| `json` (default) | Prototyping, scripts, <5k records | None |
| `sqlite` | Production, concurrent agents, large stores | None (stdlib) |

```python
# JSON
engine = ForkLedgerEngine("store.json", backend="json")

# SQLite — indexed, atomic upserts, expiry enforcement
engine = ForkLedgerEngine("store.db", backend="sqlite")
```

---

## Core concepts

### Regret vector

Computed automatically on every write and every outcome update:

```
regret(branch) = best_value_in_record − value_of_branch
```

Zero regret = the best available choice at that time. The engine uses this signal to rank branches when you request a recommendation.

### Policy distillation

When the same state pattern appears repeatedly, ForkLedger groups those forks and computes average regret per branch:

```python
engine.policies(min_support=3)
# → [{"state": {...}, "recommended_branch": "verify", "support": 7, ...}]
```

This produces a lightweight policy surface: *under this state, this branch has historically produced the lowest regret.*

### Expiry

Memories go stale. Set an expiry and ForkLedger handles the rest:

```python
ForkRecord(..., expiry="2024-12-31T00:00:00Z")
engine.purge_expired()  # or it's enforced automatically at query time
```

---

## Architecture

```
ForkLedgerEngine
├── storage/
│   ├── JsonForkStore      — flat file, zero deps, backward compat
│   └── SqliteForkStore    — indexed, atomic, production-ready
├── counterfactual.py      — regret computation
├── retrieval.py           — scoring pipeline + optional embeddings
├── policy.py              — policy distillation from repeated states
├── api.py                 — FastAPI REST layer (15 endpoints)
└── cli.py                 — full CLI (15 commands)
```

**Retrieval scoring weights:**

| Signal | Weight | Notes |
|--------|--------|-------|
| State similarity | 45% | Keyword or embedding |
| Constraint match | 20% | Hard limits |
| Recency | 15% | Decays over 30 days |
| Confidence | 10% | Record trustworthiness |
| Regret salience | 10% | High-stakes forks weighted more |

---

## Who is this for

- **Agent developers** building systems that make repeated decisions and need to improve over time
- **AI researchers** studying decision memory, counterfactual reasoning, or policy learning
- **ML engineers** who need an audit trail of their agent's choices, not just its outputs
- **Anyone** tired of agents that repeat the same mistakes

---

## Roadmap

- [ ] Fuzzy policy clustering (beyond exact state fingerprinting)
- [ ] Configurable scoring weights per domain
- [ ] LangChain + LlamaIndex + AutoGen integrations
- [ ] Data migration system for schema versioning
- [ ] Async API (asyncio + FastAPI)
- [ ] Web dashboard for decision history visualization
- [ ] Multi-agent fork sharing

---

## Contributing

Issues, PRs and ideas welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/sliper82/forkledger
cd forkledger
pip install -e ".[dev]"
pytest
```

---

## License

[Apache 2.0](LICENSE)
