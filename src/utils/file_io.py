"""File input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to a JSON Lines file.

    Args:
        path: Destination file path.
        records: Iterable of serializable dictionaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Read records from a JSON Lines file.

    Args:
        path: Source file path.

    Returns:
        Iterable of dictionaries parsed from each line.
    """
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)
