"""ForkLedger CLI — v0.3.0"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pprint import pprint

from .engine import ForkLedgerEngine
from .mcp_server import mcp_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forkledger",
        description="Branch-based decision memory for AI systems.",
    )
    parser.add_argument("--store", default=".forkledger/store.json")
    parser.add_argument("--backend", choices=["json", "sqlite"], default="json")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--half-life", type=float, default=60.0,
                        help="Confidence decay half-life in days (default: 60)")

    sub = parser.add_subparsers(dest="command", required=True)

    # add / import
    for cmd in ("add", "import"):
        p = sub.add_parser(cmd, help="Add fork records from a JSON file.")
        p.add_argument("path")

    # list
    lp = sub.add_parser("list", help="List stored fork records.")
    lp.add_argument("--limit", type=int, default=20)
    lp.add_argument("--min-confidence", type=float, default=0.0)
    lp.add_argument("--tags", nargs="*")
    lp.add_argument("--include-expired", action="store_true")

    # get
    gp = sub.add_parser("get", help="Get a single fork by ID.")
    gp.add_argument("fork_id")

    # update-outcome
    up = sub.add_parser("update-outcome", help="Update realized_value for a fork.")
    up.add_argument("fork_id")
    up.add_argument("realized_value", type=float)
    up.add_argument("--confidence", type=float, default=None)

    # delete
    dp = sub.add_parser("delete", help="Delete a fork by ID.")
    dp.add_argument("fork_id")

    # purge-expired
    sub.add_parser("purge-expired", help="Remove all expired forks.")

    # recommend
    rp = sub.add_parser("recommend", help="Recommend branches for a given state.")
    rp.add_argument("--state", required=True)
    rp.add_argument("--constraints", default="{}")
    rp.add_argument("--top-k", type=int, default=5)
    rp.add_argument("--min-score", type=float, default=0.0)
    rp.add_argument("--tags", nargs="*")
    rp.add_argument("--min-confidence", type=float, default=0.0)

    # rank
    rkp = sub.add_parser("rank", help="Rank historical forks by similarity.")
    rkp.add_argument("--state", required=True)
    rkp.add_argument("--constraints", default="{}")
    rkp.add_argument("--tags", nargs="*")

    # policies
    pp = sub.add_parser("policies", help="Distill low-regret policies.")
    pp.add_argument("--min-support", type=int, default=2)
    pp.add_argument("--min-confidence", type=float, default=0.0)
    pp.add_argument("--tags", nargs="*")
    pp.add_argument("--exact", action="store_true", help="Use exact matching instead of fuzzy")
    pp.add_argument("--threshold", type=float, default=0.6, help="Fuzzy similarity threshold")

    # win-rates
    wp = sub.add_parser("win-rates", help="Show win rate per branch across all forks.")
    wp.add_argument("--tags", nargs="*")
    wp.add_argument("--min-confidence", type=float, default=0.0)

    # accumulated-regret
    arp = sub.add_parser("accumulated-regret", help="Show CFR-style accumulated regret per branch.")
    arp.add_argument("--tags", nargs="*")
    arp.add_argument("--min-confidence", type=float, default=0.0)

    # audit
    aup = sub.add_parser("audit", help="Show decision audit trail with decay factors.")
    aup.add_argument("--limit", type=int, default=20)
    aup.add_argument("--tags", nargs="*")

    # stats
    sub.add_parser("stats", help="Show store statistics.")

    # info
    sub.add_parser("info", help="Show engine configuration.")

    # export
    ep = sub.add_parser("export", help="Export all records to a JSON file.")
    ep.add_argument("output")
    ep.add_argument("--include-expired", action="store_true")

    # serve (REST API)
    sp = sub.add_parser("serve", help="Start the REST API server.")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")

    # mcp (MCP server)
    mcp_p = sub.add_parser("mcp", help="Start the MCP server for Claude Desktop / Cursor.")
    mcp_p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    mcp_p.add_argument("--host", default="127.0.0.1")
    mcp_p.add_argument("--port", type=int, default=8001)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = args.command

    # MCP server — special case
    if cmd == "mcp":
        if not mcp_available():
            print("ERROR: Install MCP deps: pip install forkledger[mcp]", file=sys.stderr)
            sys.exit(1)
        from .mcp_server import create_mcp_server
        mcp = create_mcp_server(
            store_path=args.store,
            backend=args.backend,
            use_embeddings=args.embeddings,
        )
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport="http", host=args.host, port=args.port)
        return

    # REST API server — special case
    if cmd == "serve":
        try:
            import uvicorn
            from .api import create_app
        except ImportError:
            print("ERROR: Install API deps: pip install forkledger[api]", file=sys.stderr)
            sys.exit(1)
        app = create_app(store_path=args.store, backend=args.backend,
                         use_embeddings=args.embeddings)
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return

    engine = ForkLedgerEngine(
        store_path=args.store,
        backend=args.backend,
        use_embeddings=args.embeddings,
        confidence_half_life_days=args.half_life,
    )

    if cmd in ("add", "import"):
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
            print(f"{r.fork_id}  chosen={r.chosen_branch}  "
                  f"conf={r.confidence}  trigger={r.trigger}")

    elif cmd == "get":
        record = engine.get(args.fork_id)
        if record is None:
            print(f"not found: {args.fork_id}", file=sys.stderr); sys.exit(1)
        pprint(record.to_dict())

    elif cmd == "update-outcome":
        record = engine.update_outcome(args.fork_id, args.realized_value, args.confidence)
        if record is None:
            print(f"not found: {args.fork_id}", file=sys.stderr); sys.exit(1)
        print(f"updated: {record.fork_id}  value={record.realized_value}  "
              f"regret={record.regret_vector}")

    elif cmd == "delete":
        ok = engine.delete(args.fork_id)
        print("deleted" if ok else f"not found: {args.fork_id}")

    elif cmd == "purge-expired":
        count = engine.purge_expired()
        print(f"purged {count} expired records")

    elif cmd == "recommend":
        state = json.loads(args.state)
        constraints = json.loads(args.constraints)
        pprint(engine.recommend(state, constraints=constraints, top_k=args.top_k,
                                min_score=args.min_score, tags=args.tags,
                                min_confidence=args.min_confidence))

    elif cmd == "rank":
        state = json.loads(args.state)
        constraints = json.loads(args.constraints)
        pprint([{"fork_id": r.fork_id, "score": s, "chosen": r.chosen_branch}
                for r, s in engine.rank(state, constraints, tags=args.tags)])

    elif cmd == "policies":
        pprint(engine.policies(
            min_support=args.min_support,
            tags=args.tags,
            min_confidence=args.min_confidence,
            fuzzy=not args.exact,
            similarity_threshold=args.threshold,
        ))

    elif cmd == "win-rates":
        wr = engine.win_rates(tags=args.tags, min_confidence=args.min_confidence)
        sorted_wr = sorted(wr.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        for branch, data in sorted_wr:
            print(f"  {branch}: win_rate={data['win_rate']}  "
                  f"wins={data['wins']}/{data['appearances']}")

    elif cmd == "accumulated-regret":
        ar = engine.accumulated_regret(tags=args.tags, min_confidence=args.min_confidence)
        for branch, regret in sorted(ar.items(), key=lambda x: x[1]):
            print(f"  {branch}: {regret}")

    elif cmd == "audit":
        trail = engine.audit_trail(tags=args.tags, limit=args.limit)
        for entry in trail:
            print(f"  [{entry['created_at'][:10]}] {entry['fork_id']} "
                  f"→ '{entry['chosen_branch']}' "
                  f"outcome={entry['realized_value']} "
                  f"weight={entry['effective_weight']}")

    elif cmd == "stats":
        pprint(engine.stats())

    elif cmd == "info":
        pprint(engine.info())

    elif cmd == "export":
        records = engine.load(
            include_expired=getattr(args, "include_expired", False)
        )
        payload = [r.to_dict() for r in records]
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"exported {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
