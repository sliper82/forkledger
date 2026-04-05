"""ForkLedger CLI — upgraded with full CRUD, serve, stats, and export/import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pprint import pprint

from .engine import ForkLedgerEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forkledger",
        description="Branch-based memory engine for AI systems.",
    )
    parser.add_argument(
        "--store",
        default=".forkledger/store.json",
        help="Path to the store file (.json or .db). Default: .forkledger/store.json",
    )
    parser.add_argument(
        "--backend",
        choices=["json", "sqlite"],
        default="json",
        help="Storage backend. Use 'sqlite' for production. Default: json",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Enable semantic similarity (requires: pip install forkledger[embeddings])",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # add
    add_p = sub.add_parser("add", help="Add fork records from a JSON file.")
    add_p.add_argument("path", help="Path to a JSON array of fork records.")

    # list
    list_p = sub.add_parser("list", help="List stored fork records.")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--min-confidence", type=float, default=0.0)
    list_p.add_argument("--tags", nargs="*", help="Filter by tags.")
    list_p.add_argument("--include-expired", action="store_true")

    # get
    get_p = sub.add_parser("get", help="Get a single fork by ID.")
    get_p.add_argument("fork_id")

    # update-outcome
    upd_p = sub.add_parser("update-outcome", help="Update realized_value for an existing fork.")
    upd_p.add_argument("fork_id")
    upd_p.add_argument("realized_value", type=float)
    upd_p.add_argument("--confidence", type=float, default=None)

    # delete
    del_p = sub.add_parser("delete", help="Delete a fork by ID.")
    del_p.add_argument("fork_id")

    # purge-expired
    sub.add_parser("purge-expired", help="Remove all expired forks.")

    # recommend
    rec_p = sub.add_parser("recommend", help="Recommend branches for a given state.")
    rec_p.add_argument("--state", required=True, help="JSON object representing the current state.")
    rec_p.add_argument("--constraints", default="{}", help="JSON object of active constraints.")
    rec_p.add_argument("--top-k", type=int, default=5)
    rec_p.add_argument("--min-score", type=float, default=0.0)
    rec_p.add_argument("--tags", nargs="*")
    rec_p.add_argument("--min-confidence", type=float, default=0.0)

    # rank
    rank_p = sub.add_parser("rank", help="Rank historical forks by similarity.")
    rank_p.add_argument("--state", required=True)
    rank_p.add_argument("--constraints", default="{}")
    rank_p.add_argument("--tags", nargs="*")

    # policies
    pol_p = sub.add_parser("policies", help="Distill low-regret policies from repeated states.")
    pol_p.add_argument("--min-support", type=int, default=2)
    pol_p.add_argument("--min-confidence", type=float, default=0.0)
    pol_p.add_argument("--tags", nargs="*")

    # stats
    sub.add_parser("stats", help="Show store statistics.")

    # info
    sub.add_parser("info", help="Show engine configuration.")

    # export
    exp_p = sub.add_parser("export", help="Export all records to a JSON file.")
    exp_p.add_argument("output", help="Output JSON path.")
    exp_p.add_argument("--include-expired", action="store_true")

    # import (alias for add)
    imp_p = sub.add_parser("import", help="Import records from a JSON file (alias for add).")
    imp_p.add_argument("path", help="Path to a JSON array of fork records.")

    # serve
    srv_p = sub.add_parser("serve", help="Start the REST API server (requires: pip install forkledger[api]).")
    srv_p.add_argument("--host", default="127.0.0.1")
    srv_p.add_argument("--port", type=int, default=8000)
    srv_p.add_argument("--reload", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command

    # serve does not need an engine
    if cmd == "serve":
        try:
            import uvicorn  # type: ignore
            from .api import create_app
        except ImportError:
            print("ERROR: Install API dependencies: pip install forkledger[api]", file=sys.stderr)
            sys.exit(1)
        app = create_app(store_path=args.store, backend=args.backend, use_embeddings=args.embeddings)
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return

    engine = ForkLedgerEngine(
        store_path=args.store,
        backend=args.backend,
        use_embeddings=args.embeddings,
    )

    if cmd == "add" or cmd == "import":
        count = engine.import_json(Path(args.path))
        print(f"imported {count} fork records")

    elif cmd == "list":
        records = engine.load(
            include_expired=args.include_expired,
            tags=args.tags,
            min_confidence=args.min_confidence,
            limit=args.limit,
        )
        for r in records:
            print(f"{r.fork_id}  chosen={r.chosen_branch}  confidence={r.confidence}  trigger={r.trigger}")

    elif cmd == "get":
        record = engine.get(args.fork_id)
        if record is None:
            print(f"not found: {args.fork_id}", file=sys.stderr)
            sys.exit(1)
        pprint(record.to_dict())

    elif cmd == "update-outcome":
        record = engine.update_outcome(args.fork_id, args.realized_value, args.confidence)
        if record is None:
            print(f"not found: {args.fork_id}", file=sys.stderr)
            sys.exit(1)
        print(f"updated: {record.fork_id}  realized_value={record.realized_value}  regret={record.regret_vector}")

    elif cmd == "delete":
        ok = engine.delete(args.fork_id)
        print("deleted" if ok else f"not found: {args.fork_id}")

    elif cmd == "purge-expired":
        count = engine.purge_expired()
        print(f"purged {count} expired records")

    elif cmd == "recommend":
        state = json.loads(args.state)
        constraints = json.loads(args.constraints)
        results = engine.recommend(
            state,
            constraints=constraints,
            top_k=args.top_k,
            min_score=args.min_score,
            tags=args.tags,
            min_confidence=args.min_confidence,
        )
        pprint(results)

    elif cmd == "rank":
        state = json.loads(args.state)
        constraints = json.loads(args.constraints)
        ranked = engine.rank(state, constraints, tags=args.tags)
        pprint([
            {
                "fork_id":       r.fork_id,
                "score":         score,
                "chosen_branch": r.chosen_branch,
                "trigger":       r.trigger,
            }
            for r, score in ranked
        ])

    elif cmd == "policies":
        pprint(engine.policies(
            min_support=args.min_support,
            tags=args.tags,
            min_confidence=args.min_confidence,
        ))

    elif cmd == "stats":
        pprint(engine.stats())

    elif cmd == "info":
        pprint(engine.info())

    elif cmd == "export":
        records = engine.load(include_expired=args.include_expired)
        payload = [r.to_dict() for r in records]
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"exported {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
