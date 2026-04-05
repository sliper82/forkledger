# Changelog

## v0.2.0 — 2026-04-05

### Added
- `SqliteForkStore` — production-ready SQLite backend with indexes, atomic upserts, expiry enforcement, and stats
- `update_outcome()` — update realized value and recompute regret after the fact
- `delete()` and `purge_expired()` — full record lifecycle management
- `deduplication` — re-adding same `fork_id` updates instead of duplicating
- Expiry enforcement — expired records filtered at query time
- `load()` filters — `tags`, `min_confidence`, `limit`
- `get(fork_id)` — fetch single record by ID
- `export_json()` / `import_json()` — portability between stores
- `stats()` and `info()` — diagnostics
- REST API (`forkledger[api]`) — 15 endpoints, Swagger UI, CORS
- Semantic similarity (`forkledger[embeddings]`) — via sentence-transformers, graceful fallback
- `min_score` threshold for recommendations
- CLI: 15 commands including `serve`, `export`, `import`, `update-outcome`, `delete`, `purge-expired`
- `--backend` and `--embeddings` CLI flags
- GitHub Actions CI — Python 3.10/3.11/3.12, lint, build, auto-publish to PyPI on tag
- 37 tests (up from 2), both backends covered

### Changed
- `pyproject.toml` optional dependency groups: `[api]`, `[embeddings]`, `[all]`, `[dev]`
- Version bumped to `0.2.0`

## v0.1.0 — initial release

- Core `ForkRecord` model
- `JsonForkStore`
- Regret computation
- Keyword-based retrieval and scoring
- Policy distillation
- Basic CLI (5 commands)

## v0.3.0 — 2026-04-05

### Added
- `mcp_server.py`: Full MCP server with 8 tools — works with Claude Desktop, Cursor, VS Code, AutoGen
- `policy.py`: Fuzzy policy clustering (greedy state similarity grouping, configurable threshold)
- `policy.py`: Branch win rates embedded in policy output
- `policy.py`: Policy confidence score based on consistency + support
- `counterfactual.py`: `confidence_decay_factor()` — exponential time-decay weighting
- `counterfactual.py`: `accumulate_regret()` — CFR-style weighted regret accumulation
- `counterfactual.py`: `branch_win_rate()` — win rate per branch across all records
- `counterfactual.py`: `normalized_regret()` — cross-record normalized regret
- `engine.py`: `accumulated_regret()`, `win_rates()`, `decay_factor()`, `audit_trail()`
- `engine.py`: `confidence_half_life_days` parameter for decay control
- `storage_pg.py`: PostgreSQL + pgvector backend with GIN indexes, array tags, JSONB
- `dashboard/app.py`: Streamlit web UI with 6 pages — overview, decisions, recommendations, policies, analytics, audit
- `Dockerfile` + `docker-compose.yml`: one-command deployment
- `benchmarks/locomo_benchmark.py`: decision retrieval benchmark (Precision@1, latency p50/p95, policy accuracy)
- `COMPARISON.md`: ForkLedger vs Mem0, Hindsight, LangMem, mcp-memory
- CLI: `win-rates`, `accumulated-regret`, `audit`, `mcp` commands; `--half-life` flag
- pyproject.toml: `[mcp]`, `[postgres]`, `[dashboard]` optional dep groups; v0.3.0
- GitHub topics: ai, memory, agents, mcp, decision-engine, counterfactuals, regret, sqlite

### Changed
- `policy.py`: `distill_policies()` now defaults to fuzzy=True
- `engine.py`: `policies()` exposes `fuzzy` and `similarity_threshold` params
- `engine.py`: `stats()` now includes top winning branch
- `engine.py`: `info()` now returns version and half-life config
- README: complete rewrite targeting viral GitHub discovery
