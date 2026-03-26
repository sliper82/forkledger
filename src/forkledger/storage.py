from __future__ import annotations

import json
from pathlib import Path

from .models import ForkRecord


class JsonForkStore:
    def __init__(self, path: str | Path = ".forkledger/store.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> list[ForkRecord]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [ForkRecord.from_dict(item) for item in payload]

    def save(self, records: list[ForkRecord]) -> None:
        payload = [record.to_dict() for record in records]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, records: list[ForkRecord]) -> None:
        existing = self.load()
        existing.extend(records)
        self.save(existing)
