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
