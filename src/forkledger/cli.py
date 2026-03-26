from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint

from .engine import ForkLedgerEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forkledger", description="Branch-based memory for AI systems.")
    parser.add_argument("--store", default=".forkledger/store.json", help="Path to the local ForkLedger store.")

    sub = parser.add_subparsers(dest="command", required=True)

    add_parser = sub.add_parser("add", help="Add fork records from a JSON file.")
    add_parser.add_argument("path", help="Path to a JSON array of fork records.")

    list_parser = sub.add_parser("list", help="List stored fork records.")
    list_parser.add_argument("--limit", type=int, default=20)

    rec_parser = sub.add_parser("recommend", help="Recommend branches for a given state.")
    rec_parser.add_argument("--state", required=True, help="JSON object representing the current state.")
    rec_parser.add_argument("--constraints", default="{}", help="JSON object of active constraints.")
    rec_parser.add_argument("--top-k", type=int, default=5)

    rank_parser = sub.add_parser("rank", help="Rank historical forks by similarity.")
    rank_parser.add_argument("--state", required=True)
    rank_parser.add_argument("--constraints", default="{}")

    pol_parser = sub.add_parser("policies", help="Distill low-regret policies from repeated states.")
    pol_parser.add_argument("--min-support", type=int, default=2)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    engine = ForkLedgerEngine(args.store)

    if args.command == "add":
        payload = engine.load_payload_file(Path(args.path))
        records = engine.add_records_from_payload(payload)
        print(f"added {len(records)} fork records")

    elif args.command == "list":
        records = engine.load()[: args.limit]
        for record in records:
            print(f"{record.fork_id} :: chosen={record.chosen_branch} :: trigger={record.trigger}")

    elif args.command == "recommend":
        state = json.loads(args.state)
        constraints = json.loads(args.constraints)
        pprint(engine.recommend(state, constraints=constraints, top_k=args.top_k))

    elif args.command == "rank":
        state = json.loads(args.state)
        constraints = json.loads(args.constraints)
        ranked = engine.rank(state, constraints)
        pprint(
            [
                {
                    "fork_id": record.fork_id,
                    "score": score,
                    "chosen_branch": record.chosen_branch,
                    "trigger": record.trigger,
                }
                for record, score in ranked
            ]
        )

    elif args.command == "policies":
        pprint(engine.policies(min_support=args.min_support))


if __name__ == "__main__":
    main()
