"""ForkLedger REST API — powered by FastAPI.

Run with:
    pip install forkledger[api]
    forkledger serve

Or directly:
    uvicorn forkledger.api:app --reload
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from .engine import ForkLedgerEngine
from .models import Branch, ForkRecord, OutcomeEstimate

# ---------------------------------------------------------------------------
# Pydantic schemas (only defined when fastapi is installed)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

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

    class RecommendRequest(BaseModel):
        current_state: dict[str, Any]
        constraints: dict[str, Any] = Field(default_factory=dict)
        top_k: int = 5
        min_score: float = 0.0
        tags: list[str] | None = None
        min_confidence: float = 0.0

    class OutcomeUpdateRequest(BaseModel):
        realized_value: float
        confidence: float | None = None

    # -----------------------------------------------------------------------
    # App factory
    # -----------------------------------------------------------------------

    def create_app(
        store_path: str = ".forkledger/store.db",
        backend: str = "sqlite",
        use_embeddings: bool = False,
    ) -> FastAPI:
        engine = ForkLedgerEngine(
            store_path=store_path,
            backend=backend,  # type: ignore[arg-type]
            use_embeddings=use_embeddings,
        )

        app = FastAPI(
            title="ForkLedger API",
            description="Branch-based memory engine for AI systems.",
            version="0.2.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # -------------------------------------------------------------------
        # Health & info
        # -------------------------------------------------------------------

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/info")
        def info() -> dict[str, Any]:
            return engine.info()

        @app.get("/stats")
        def stats() -> dict[str, Any]:
            return engine.stats()

        # -------------------------------------------------------------------
        # Forks CRUD
        # -------------------------------------------------------------------

        @app.get("/forks")
        def list_forks(
            limit: int = Query(50, ge=1, le=1000),
            min_confidence: float = Query(0.0, ge=0.0, le=1.0),
            tags: list[str] | None = Query(None),
            include_expired: bool = False,
        ) -> list[dict[str, Any]]:
            records = engine.load(
                include_expired=include_expired,
                tags=tags,
                min_confidence=min_confidence,
                limit=limit,
            )
            return [r.to_dict() for r in records]

        @app.get("/forks/{fork_id}")
        def get_fork(fork_id: str) -> dict[str, Any]:
            record = engine.get(fork_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Fork '{fork_id}' not found.")
            return record.to_dict()

        @app.post("/forks", status_code=201)
        def add_fork(payload: ForkRecordIn) -> dict[str, Any]:
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
            )
            result = engine.add_record(record)
            return result.to_dict()

        @app.post("/forks/bulk", status_code=201)
        def bulk_add_forks(payload: list[ForkRecordIn]) -> dict[str, Any]:
            raw = [item.model_dump() for item in payload]
            records = engine.add_records_from_payload(raw)
            return {"added": len(records)}

        @app.patch("/forks/{fork_id}/outcome")
        def update_outcome(fork_id: str, body: OutcomeUpdateRequest) -> dict[str, Any]:
            record = engine.update_outcome(fork_id, body.realized_value, body.confidence)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Fork '{fork_id}' not found.")
            return record.to_dict()

        @app.delete("/forks/{fork_id}")
        def delete_fork(fork_id: str) -> dict[str, Any]:
            ok = engine.delete(fork_id)
            if not ok:
                raise HTTPException(status_code=404, detail=f"Fork '{fork_id}' not found.")
            return {"deleted": fork_id}

        @app.delete("/forks/expired")
        def purge_expired() -> dict[str, Any]:
            count = engine.purge_expired()
            return {"purged": count}

        # -------------------------------------------------------------------
        # Retrieval
        # -------------------------------------------------------------------

        @app.post("/recommend")
        def recommend(body: RecommendRequest) -> list[dict[str, Any]]:
            return engine.recommend(
                current_state=body.current_state,
                constraints=body.constraints,
                top_k=body.top_k,
                min_score=body.min_score,
                tags=body.tags,
                min_confidence=body.min_confidence,
            )

        @app.post("/rank")
        def rank(body: RecommendRequest) -> list[dict[str, Any]]:
            ranked = engine.rank(
                current_state=body.current_state,
                constraints=body.constraints,
                tags=body.tags,
                min_confidence=body.min_confidence,
            )
            return [
                {
                    "fork_id":        r.fork_id,
                    "score":          score,
                    "chosen_branch":  r.chosen_branch,
                    "trigger":        r.trigger,
                }
                for r, score in ranked
            ]

        @app.get("/policies")
        def policies(
            min_support: int = Query(2, ge=1),
            min_confidence: float = Query(0.0, ge=0.0, le=1.0),
            tags: list[str] | None = Query(None),
        ) -> list[dict[str, Any]]:
            return engine.policies(
                min_support=min_support,
                tags=tags,
                min_confidence=min_confidence,
            )

        # -------------------------------------------------------------------
        # Export / import
        # -------------------------------------------------------------------

        @app.get("/export")
        def export_all() -> list[dict[str, Any]]:
            records = engine.load(include_expired=True)
            return [r.to_dict() for r in records]

        return app

    # Default app instance (for `uvicorn forkledger.api:app`)
    app = create_app()
