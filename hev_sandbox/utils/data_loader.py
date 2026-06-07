"""Data loading utilities for JSON, JSONL, JSONL.GZ, and text corpora."""

from __future__ import annotations

import gzip
import json
from typing import Dict, Iterator, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def iter_jsonl(file_path: str | Path) -> Iterator[Dict]:
    file_path = Path(file_path)
    with _open_text(file_path, "rt") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Error parsing line %s in %s: %s", line_num, file_path, exc)


def load_jsonl(file_path: str | Path, limit: Optional[int] = None) -> List[Dict]:
    """Load data from a JSONL or JSONL.GZ file."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return []
    data = []
    for item in iter_jsonl(file_path):
        data.append(item)
        if limit is not None and len(data) >= limit:
            break
    logger.info("Loaded %s records from %s", len(data), file_path)
    return data


def save_jsonl(data: List[Dict], file_path: str | Path) -> int:
    """Save data to a JSONL or JSONL.GZ file and return number of records."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(file_path, "wt") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %s records to %s", len(data), file_path)
    return len(data)


def load_json(file_path: str | Path) -> Dict:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, file_path: str | Path, indent: int = 2) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info("Saved data to %s", file_path)


def load_text_corpus(file_path: str | Path, max_lines: int = None) -> List[str]:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return []
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    logger.info("Loaded %s lines from %s", len(lines), file_path)
    return lines
