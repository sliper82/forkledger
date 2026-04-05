"""
ForkLedger REST API — v0.4.0 (async FastAPI)

New in this version:
- Full async/await throughout (concurrent request handling)
- /forks/search endpoint for FTS5 full-text search
- /recommend supports domain-specific scoring weights
- /win-rates and /audit endpoints
- /namespaces endpoint for multi-agent inspection
- namespace filter on all retrieval endpoints
- outcome_source field on PATCH /outcome
- Webhook support: POST outcome update fires webhook if configured
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import httpx
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from .engine import ForkLedgerEngine
from .models import Branch, ForkRecord, OutcomeEstimate
from .retrieval import ScoringWeights

if _FASTAPI_AVAILABLE:

    # ── Pydantic schemas ────────────────────────────────────────────────────

    class BranchIn(BaseModel):
        name: str
        description: str = ""
        metadata: dict[str, Any] = Field(default_factory=dict)

    class OutcomeEstimateIn(BaseModel):
        branch_name: str
        estimated_value: float
        confidence: float = 0.5
        rationale: str = ""

    class ForkRecordIn(BaseModel):
        fork_id: str
        pre_state: dict[str, Any]
        trigger: str
        possible_branches: list[BranchIn]
        chosen_branch: str
        realized_value: float
        estimated_outcomes: list[OutcomeEstimateIn] = Field(default_factory=list)
        constraints: dict[str, Any] = Field(default_factory=dict)
        provenance: dict[str, Any] = Field(default_factory=dict)
        confidence: float = 0.5
        expiry: str | None = None
        tags: list[str] = Field(default_factory=list)
        namespace: str = "default"
        outcome_source: str = "observed"

    class OutcomeUpdateRequest(BaseModel):
        realized_value: float
        confidence: float | None = None
        outcome_source: str = "observed"

    class RecommendRequest(BaseModel):
        current_state: dict[str, Any]
        constraints: dict[str, Any] = Field(default_factory=dict)
        top_k: int = 5
        min_score: float = 0.0
        tags: list[str] | None = None
        min_confidence: float = 0.0
        namespace: str | None = None
        domain: str = "default"  # for pre-tuned scoring weights

    # ── Webhook helper ──────────────────────────────────────────────────────

    async def _fire_webhook(url: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget webhook notification."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json=payload)
        except Exception:
            pass  # webhook failures are non-fatal

    # ── App factory ─────────────────────────────────────────────────────────

    def create_app(
        store_path: str = ".forkledger/store.db",
        backend: str = "sqlite",
        use_embeddings: bool = False,
        webhook_url: str | None = None,
    ) -> FastAPI:
        engine = ForkLedgerEngine(
            store_path=store_path,
            backend=backend,  # type: ignore
            use_embeddings=use_embeddings,
        )
        _webhook_url = webhook_url or os.getenv("FORKLEDGER_WEBHOOK_URL")

        app = FastAPI(
            title="ForkLedger API",
            description=(
                "Branch-based decision memory for AI agents.\n\n"
                "Stores decisions with alternatives, outcomes, and regret — "
                "learns which branches consistently produce the lowest regret."
            ),
            version="0.4.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Health & info ───────────────────────────────────────────────────

        @app.get("/health", tags=["system"])
        async def health() -> dict[str, str]:
            return {"status": "ok", "version": "0.4.0"}

        @app.get("/info", tags=["system"])
        async def info() -> dict[str, Any]:
            return await asyncio.get_event_loop().run_in_executor(None, engine.info)

        @app.get("/stats", tags=["system"])
        async def stats() -> dict[str, Any]:
            return await asyncio.get_event_loop().run_in_executor(None, engine.stats)

        @app.get("/namespaces", tags=["system"])
        async def list_namespaces() -> list[dict[str, Any]]:
            s = await asyncio.get_event_loop().run_in_executor(None, engine.stats)
            return s.get("namespaces", [])

        # ── Forks CRUD ──────────────────────────────────────────────────────

        @app.get("/forks", tags=["forks"])
        async def list_forks(
            limit: int = Query(50, ge=1, le=1000),
            min_confidence: float = Query(0.0, ge=0.0, le=1.0),
            tags: list[str] | None = Query(None),
            namespace: str | None = None,
            include_expired: bool = False,
        ) -> list[dict[str, Any]]:
            def _load():
                return engine.load(
                    include_expired=include_expired,
                    tags=tags,
                    min_confidence=min_confidence,
                    limit=limit,
                    namespace=namespace,
                )
            records = await asyncio.get_event_loop().run_in_executor(None, _load)
            return [r.to_dict() for r in records]

        @app.get("/forks/search", tags=["forks"])
        async def search_forks(
            q: str = Query(..., description="FTS5 query string"),
            limit: int = Query(20, ge=1, le=200),
            namespace: str | None = None,
        ) -> list[dict[str, Any]]:
            """Full-text search across trigger, state, branch names, and tags."""
            from .storage import SqliteForkStore
            if not isinstance(engine.store, SqliteForkStore):
                raise HTTPException(400, "Full-text search requires SQLite backend")
            def _search():
                return engine.store.search_text(q, limit=limit)
            records = await asyncio.get_event_loop().run_in_executor(None, _search)
            if namespace:
                records = [r for r in records if r.namespace == namespace]
            return [r.to_dict() for r in records]

        @app.get("/forks/{fork_id}", tags=["forks"])
        async def get_fork(fork_id: str) -> dict[str, Any]:
            record = await asyncio.get_event_loop().run_in_executor(
                None, engine.get, fork_id
            )
            if record is None:
                raise HTTPException(404, f"Fork '{fork_id}' not found")
            return record.to_dict()

        @app.post("/forks", status_code=201, tags=["forks"])
        async def add_fork(payload: ForkRecordIn) -> dict[str, Any]:
            record = ForkRecord(
                fork_id=payload.fork_id,
                pre_state=payload.pre_state,
                trigger=payload.trigger,
                possible_branches=[Branch(**b.model_dump()) for b in payload.possible_branches],
                chosen_branch=payload.chosen_branch,
                realized_value=payload.realized_value,
                estimated_outcomes=[OutcomeEstimate(**e.model_dump()) for e in payload.estimated_outcomes],
                constraints=payload.constraints,
                provenance=payload.provenance,
                confidence=payload.confidence,
                expiry=payload.expiry,
                tags=payload.tags,
                namespace=payload.namespace,
                outcome_source=payload.outcome_source,
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, engine.add_record, record
            )
            return result.to_dict()

        @app.post("/forks/bulk", status_code=201, tags=["forks"])
        async def bulk_add_forks(payload: list[ForkRecordIn]) -> dict[str, Any]:
            raw = [item.model_dump() for item in payload]
            def _bulk():
                return engine.add_records_from_payload(raw)
            records = await asyncio.get_event_loop().run_in_executor(None, _bulk)
            return {"added": len(records)}

        @app.patch("/forks/{fork_id}/outcome", tags=["forks"])
        async def update_outcome(
            fork_id: str,
            body: OutcomeUpdateRequest,
            background_tasks: BackgroundTasks,
        ) -> dict[str, Any]:
            def _update():
                return engine.update_outcome(
                    fork_id, body.realized_value, body.confidence, body.outcome_source
                )
            record = await asyncio.get_event_loop().run_in_executor(None, _update)
            if record is None:
                raise HTTPException(404, f"Fork '{fork_id}' not found")

            # Fire webhook if configured (non-blocking)
            if _webhook_url:
                background_tasks.add_task(
                    _fire_webhook,
                    _webhook_url,
                    {"event": "outcome_updated", "fork_id": fork_id,
                     "realized_value": body.realized_value,
                     "regret_vector": record.regret_vector},
                )
            return record.to_dict()

        @app.delete("/forks/{fork_id}", tags=["forks"])
        async def delete_fork(fork_id: str) -> dict[str, Any]:
            ok = await asyncio.get_event_loop().run_in_executor(
                None, engine.delete, fork_id
            )
            if not ok:
                raise HTTPException(404, f"Fork '{fork_id}' not found")
            return {"deleted": fork_id}

        @app.delete("/forks/expired", tags=["forks"])
        async def purge_expired() -> dict[str, Any]:
            count = await asyncio.get_event_loop().run_in_executor(
                None, engine.purge_expired
            )
            return {"purged": count}

        # ── Retrieval ───────────────────────────────────────────────────────

        @app.post("/recommend", tags=["retrieval"])
        async def recommend(body: RecommendRequest) -> list[dict[str, Any]]:
            weights = ScoringWeights.for_domain(body.domain)
            def _rec():
                return engine.recommend(
                    current_state=body.current_state,
                    constraints=body.constraints,
                    top_k=body.top_k,
                    min_score=body.min_score,
                    tags=body.tags,
                    min_confidence=body.min_confidence,
                    namespace=body.namespace,
                    weights=weights,
                )
            return await asyncio.get_event_loop().run_in_executor(None, _rec)

        @app.post("/rank", tags=["retrieval"])
        async def rank(body: RecommendRequest) -> list[dict[str, Any]]:
            weights = ScoringWeights.for_domain(body.domain)
            def _rank():
                return engine.rank(
                    body.current_state, body.constraints,
                    tags=body.tags, min_confidence=body.min_confidence,
                    namespace=body.namespace, weights=weights,
                )
            ranked = await asyncio.get_event_loop().run_in_executor(None, _rank)
            return [
                {"fork_id": r.fork_id, "score": s,
                 "chosen_branch": r.chosen_branch, "trigger": r.trigger}
                for r, s in ranked
            ]

        @app.get("/policies", tags=["retrieval"])
        async def policies(
            min_support: int = Query(2, ge=1),
            min_confidence: float = Query(0.0, ge=0.0, le=1.0),
            tags: list[str] | None = Query(None),
            namespace: str | None = None,
            fuzzy: bool = True,
            threshold: float = Query(0.6, ge=0.0, le=1.0),
        ) -> list[dict[str, Any]]:
            def _pol():
                return engine.policies(
                    min_support=min_support, tags=tags,
                    min_confidence=min_confidence, namespace=namespace,
                    fuzzy=fuzzy, similarity_threshold=threshold,
                )
            return await asyncio.get_event_loop().run_in_executor(None, _pol)

        @app.get("/win-rates", tags=["analytics"])
        async def win_rates(
            tags: list[str] | None = Query(None),
            min_confidence: float = Query(0.0),
            namespace: str | None = None,
        ) -> dict[str, Any]:
            def _wr():
                return engine.win_rates(tags=tags, min_confidence=min_confidence,
                                        namespace=namespace)
            return await asyncio.get_event_loop().run_in_executor(None, _wr)

        @app.get("/accumulated-regret", tags=["analytics"])
        async def accumulated_regret(
            tags: list[str] | None = Query(None),
            namespace: str | None = None,
        ) -> dict[str, float]:
            def _ar():
                return engine.accumulated_regret(tags=tags, namespace=namespace)
            return await asyncio.get_event_loop().run_in_executor(None, _ar)

        @app.get("/audit", tags=["analytics"])
        async def audit(
            limit: int = Query(50, ge=1, le=500),
            tags: list[str] | None = Query(None),
            namespace: str | None = None,
        ) -> list[dict[str, Any]]:
            def _audit():
                return engine.audit_trail(tags=tags, limit=limit, namespace=namespace)
            return await asyncio.get_event_loop().run_in_executor(None, _audit)

        @app.get("/export", tags=["system"])
        async def export_all() -> list[dict[str, Any]]:
            records = await asyncio.get_event_loop().run_in_executor(
                None, lambda: engine.load(include_expired=True)
            )
            return [r.to_dict() for r in records]

        return app

    app = create_app()
