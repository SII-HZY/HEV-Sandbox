#!/usr/bin/env python
"""Aggregate HEV detail JSONL result files into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hev_sandbox.utils.data_loader import iter_jsonl  # noqa: E402


def _detail_files_from_path(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        unified = sorted(path.rglob("detail.jsonl")) + sorted(path.rglob("detail.jsonl.gz"))
        unified_parents = {file.parent for file in unified}
        per_user = sorted(path.rglob("*_detail.jsonl")) + sorted(path.rglob("*_detail.jsonl.gz"))
        # A run directory contains both a unified detail.jsonl and per-user
        # detail files. Prefer the unified file to avoid double-counting.
        per_user = [file for file in per_user if file.parent not in unified_parents]
        return unified + per_user
    return []


def load_detail_files(paths: Iterable[Path]) -> List[dict]:
    files: List[Path] = []
    seen = set()
    for path in paths:
        for file in _detail_files_from_path(path):
            if file not in seen:
                files.append(file)
                seen.add(file)
    rows = []
    for path in files:
        for row in iter_jsonl(path):
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def _safety(row: dict) -> str:
    return row.get("safety") or (row.get("evaluation") or {}).get("safety") or "undetermined"


def metric(rows: List[dict]) -> dict:
    total = len(rows)
    safe = sum(1 for r in rows if _safety(r) == "safe")
    unsafe = sum(1 for r in rows if _safety(r) == "unsafe")
    und = sum(1 for r in rows if _safety(r) == "undetermined")
    return {"total": total, "safe": safe, "unsafe": unsafe, "undetermined": und, "lef": unsafe / total if total else 0.0}


def _value(row: dict, key: str):
    if key in row and row.get(key) is not None:
        return row.get(key)
    scenario = row.get("scenario") or {}
    if key in scenario and scenario.get(key) is not None:
        return scenario.get(key)
    evaluation = row.get("evaluation") or {}
    if key in evaluation and evaluation.get(key) is not None:
        return evaluation.get(key)
    if key == "hazard_name":
        return row.get("hazard") or scenario.get("hazard") or evaluation.get("hazard") or ""
    return ""


def write_grouped(rows: List[dict], output_path: Path, keys: List[str]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(_value(row, k) for k in keys)].append(row)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(keys) + ["total", "safe", "unsafe", "undetermined", "lef", "lef_percent"])
        writer.writeheader()
        for group_key, items in sorted(groups.items()):
            m = metric(items)
            writer.writerow({
                **{key: value for key, value in zip(keys, group_key)},
                "total": m["total"],
                "safe": m["safe"],
                "unsafe": m["unsafe"],
                "undetermined": m["undetermined"],
                "lef": round(m["lef"], 4),
                "lef_percent": round(m["lef"] * 100, 2),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate HEV result detail files")
    parser.add_argument("paths", nargs="*", help="Optional result files or directories. Defaults to --input_dir.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", default="results")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="results/aggregated")
    args = parser.parse_args()
    input_paths = [Path(p) for p in args.paths] if args.paths else [Path(args.input_dir)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_detail_files(input_paths)
    with open(output_dir / "overall.json", "w", encoding="utf-8") as handle:
        json.dump(metric(rows), handle, ensure_ascii=False, indent=2)
    write_grouped(rows, output_dir / "by_domain.csv", ["domain"])
    write_grouped(rows, output_dir / "by_user_type.csv", ["user_type"])
    write_grouped(rows, output_dir / "by_hazard.csv", ["hazard_name"])
    write_grouped(rows, output_dir / "by_domain_user_type.csv", ["domain", "user_type"])
    print(f"Aggregated {len(rows)} rows into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
