from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .counterfactual import fill_regret
from .models import ForkRecord
from .policy import distill_policies
from .retrieval import rank_records, recommend_branches
from .storage import JsonForkStore


class ForkLedgerEngine:
    def __init__(self, store_path: str | Path = ".forkledger/store.json") -> None:
        self.store = JsonForkStore(store_path)

    def add_record(self, record: ForkRecord) -> ForkRecord:
        filled = fill_regret(record)
        self.store.append([filled])
        return filled

    def add_records_from_payload(self, payload: list[dict[str, Any]]) -> list[ForkRecord]:
        records = [fill_regret(ForkRecord.from_dict(item)) for item in payload]
        self.store.append(records)
        return records

    def load(self) -> list[ForkRecord]:
        return self.store.load()

    def recommend(self, current_state: dict[str, Any], constraints: dict[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        records = self.load()
        return recommend_branches(records, current_state=current_state, constraints=constraints, top_k=top_k)

    def rank(self, current_state: dict[str, Any], constraints: dict[str, Any] | None = None) -> list[tuple[ForkRecord, float]]:
        records = self.load()
        return rank_records(records, current_state=current_state, constraints=constraints)

    def policies(self, min_support: int = 2) -> list[dict[str, Any]]:
        return distill_policies(self.load(), min_support=min_support)

    @staticmethod
    def load_payload_file(path: str | Path) -> list[dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
