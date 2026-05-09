"""IO helpers for JSON/JSONL files used throughout the pipeline."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable, Iterator, List


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield JSON records from a JSON Lines file. Skips malformed lines."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def append_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Append records to a JSON Lines file. Returns the number of lines written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def clear_cuda_cache() -> None:
    """Best-effort GPU memory cleanup. Safe to call even on CPU-only systems."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def sets_to_sorted_lists(d):
    """Recursively convert sets to sorted lists for JSON serialization."""
    if isinstance(d, dict):
        return {k: sets_to_sorted_lists(v) for k, v in d.items()}
    if isinstance(d, set):
        return sorted(int(x) for x in d)
    if isinstance(d, (list, tuple)):
        return [sets_to_sorted_lists(x) for x in d]
    return d


def lists_to_sets(d):
    """Recursively convert leaf int lists back to sets after JSON round-trip."""
    if isinstance(d, dict):
        return {k: lists_to_sets(v) for k, v in d.items()}
    if isinstance(d, list) and all(isinstance(x, int) for x in d):
        return set(d)
    if isinstance(d, list):
        return [lists_to_sets(x) for x in d]
    return d


def first_existing(paths: List[str | Path]) -> Path:
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"None of these paths exist: {paths}")
