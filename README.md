# ForkLedger

**A branch-based memory engine for AI systems that learns from decisions, alternatives, and regret.**

---

Most AI memory systems answer: *"What happened?"*

ForkLedger answers: *"When this situation appeared before — what options existed, which path was chosen, and what did that choice cost?"*

---

## Why ForkLedger

Traditional memory stores messages, chunks, or embeddings.  
ForkLedger stores **decision geometry**: the state before a choice, the available branches, the path taken, the observed result, and the estimated regret of every alternative.

Over time, the engine learns which branches consistently lead to lower regret under similar conditions — and surfaces that knowledge when it matters.

---

## Installation

```bash
# Core (zero dependencies)
pip install forkledger

# With REST API
pip install forkledger[api]

# With semantic similarity (sentence-transformers)
pip install forkledger[embeddings]

# Everything
pip install forkledger[all]
```

---

## Quickstart

```python
from forkledger import ForkLedgerEngine, ForkRecord, Branch, OutcomeEstimate

engine = ForkLedgerEngine(".forkledger/store.db", backend="sqlite")

record = ForkRecord(
    fork_id="trade-001",
    pre_state={"market": "BTC", "signal": "breakout", "volume": "high"},
    trigger="RSI crossed 70 with volume confirmation",
    possible_branches=[Branch(name="enter"), Branch(name="wait"), Branch(name="short")],
    chosen_branch="enter",
    realized_value=2.4,
    estimated_outcomes=[
        OutcomeEstimate(branch_name="wait", estimated_value=0.0),
        OutcomeEstimate(branch_name="short", estimated_value=-1.5),
    ],
    confidence=0.8,
    tags=["crypto", "momentum"],
)

engine.add_record(record)

# Later: update with actual outcome
engine.update_outcome("trade-001", realized_value=1.9, confidence=0.85)

# Recommend branches for a new similar situation
recs = engine.recommend(
    current_state={"market": "ETH", "signal": "breakout", "volume": "high"},
    constraints={"risk": "medium"},
    top_k=3,
)

for rec in recs:
    print(rec["branch"], rec["score"])
```

---

## Core Concepts

### Fork Record

The atomic unit of memory. Each fork captures:

| Field | Description |
|---|---|
| `pre_state` | The world before the decision |
| `trigger` | Why this decision was consequential |
| `possible_branches` | All available options |
| `chosen_branch` | What was actually done |
| `realized_value` | Observed outcome (updatable) |
| `estimated_outcomes` | Counterfactual estimates for alternatives |
| `regret_vector` | Opportunity cost per branch (auto-computed) |
| `constraints` | Active hard limits at decision time |
| `confidence` | Record trustworthiness (0–1) |
| `expiry` | Optional invalidation timestamp |
| `tags` | Filterable labels |

### Regret Vector

ForkLedger computes regret automatically on every write and update:

```
regret(branch) = best_value_in_record - value_of_branch
```

Zero regret = best available choice. The engine uses this to rank branches when you request a recommendation.

### Policy Distillation

When the same state pattern appears multiple times, ForkLedger groups those forks and computes average regret per branch. The result is a lightweight policy surface: under this state, this branch has historically produced the lowest regret.

---

## Storage Backends

### JSON (default, zero dependencies)

```python
engine = ForkLedgerEngine(".forkledger/store.json", backend="json")
```

Good for: prototyping, local scripts, <5 000 records.

### SQLite (recommended for production)

```python
engine = ForkLedgerEngine(".forkledger/store.db", backend="sqlite")
```

Good for: any serious workload. Adds indexed queries, atomic upserts, expiry enforcement, and rich stats.

---

## Semantic Similarity (Optional)

Install the embeddings extra to enable cosine similarity retrieval via `sentence-transformers`:

```bash
pip install forkledger[embeddings]
```

```python
engine = ForkLedgerEngine(
    ".forkledger/store.db",
    backend="sqlite",
    use_embeddings=True,
    embedding_model="all-MiniLM-L6-v2",
)
```

If `sentence-transformers` is not installed, ForkLedger falls back silently to keyword overlap scoring.

---

## REST API

Requires: `pip install forkledger[api]`

### Start the server

```bash
forkledger serve --backend sqlite --store .forkledger/store.db --port 8000
```

Or with uvicorn directly:

```bash
uvicorn forkledger.api:app --reload
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/info` | Engine configuration |
| `GET` | `/stats` | Store statistics |
| `GET` | `/forks` | List forks (filterable) |
| `GET` | `/forks/{id}` | Get single fork |
| `POST` | `/forks` | Add one fork |
| `POST` | `/forks/bulk` | Bulk add forks |
| `PATCH` | `/forks/{id}/outcome` | Update realized value |
| `DELETE` | `/forks/{id}` | Delete fork |
| `DELETE` | `/forks/expired` | Purge all expired |
| `POST` | `/recommend` | Recommend branches |
| `POST` | `/rank` | Rank forks by similarity |
| `GET` | `/policies` | Distilled low-regret policies |
| `GET` | `/export` | Export all records as JSON |

---

## CLI Reference

```bash
# Add records from file
forkledger add records.json

# List records
forkledger list --limit 20 --tags crypto --min-confidence 0.7

# Get single record
forkledger get trade-001

# Update outcome after the fact
forkledger update-outcome trade-001 1.9 --confidence 0.85

# Recommend branches for current state
forkledger recommend --state '{"signal": "breakout", "volume": "high"}' --top-k 3

# View distilled policies
forkledger policies --min-support 3

# Stats
forkledger stats

# Remove expired records
forkledger purge-expired

# Export / import
forkledger export backup.json
forkledger import backup.json

# Start API server
forkledger serve --backend sqlite --port 8000

# Use SQLite backend
forkledger --backend sqlite --store store.db list
```

---

## Python API Reference

```python
# Engine init
engine = ForkLedgerEngine(store_path, backend="json"|"sqlite", use_embeddings=False)

# Write
engine.add_record(record)                          # single record
engine.add_records_from_payload(list_of_dicts)     # bulk from dicts

# Update
engine.update_outcome(fork_id, realized_value, confidence=None)

# Read
engine.load(include_expired, tags, min_confidence, limit)
engine.get(fork_id)

# Delete
engine.delete(fork_id)
engine.purge_expired()

# Retrieval
engine.recommend(current_state, constraints, top_k, min_score, tags, min_confidence)
engine.rank(current_state, constraints, tags, min_confidence)
engine.policies(min_support, tags, min_confidence)

# Diagnostics
engine.stats()
engine.info()

# Import / export
engine.export_json(path)
engine.import_json(path)
```

---

## Architecture

```
ForkLedgerEngine
├── storage/
│   ├── JsonForkStore      — flat file, zero deps
│   └── SqliteForkStore    — indexed, production-ready
├── counterfactual.py      — regret computation
├── retrieval.py           — scoring + embedding similarity
├── policy.py              — policy distillation from repeated states
├── api.py                 — FastAPI REST layer
└── cli.py                 — command-line interface
```

Scoring weights (retrieval):

| Signal | Weight |
|---|---|
| State similarity | 45% |
| Constraint match | 20% |
| Recency | 15% |
| Confidence | 10% |
| Regret salience | 10% |

---

## Extension Points

- **Learned state encoders** — replace keyword overlap with domain-trained embeddings
- **Simulator-backed counterfactuals** — run alternatives in simulation before committing
- **Multi-agent fork sharing** — agents learn from each other's decisions
- **Time-decay policies** — old evidence loses weight automatically
- **Webhook on outcome update** — notify external systems when regret is recomputed
- **Evaluation dashboards** — visualize decision history and policy confidence

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

Tests: `pytest`  
Lint: `ruff check src/`

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
