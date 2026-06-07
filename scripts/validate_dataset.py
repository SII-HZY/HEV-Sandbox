#!/usr/bin/env python
"""Validate normalized HEV benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hev_sandbox.dataset import load_benchmark_split, validate_rows  # noqa: E402
from hev_sandbox.utils import save_json  # noqa: E402


def parse_domains(values):
    if not values:
        return None
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    out = []
    for value in values:
        out.extend(part.strip() for part in str(value).replace(",", " ").split() if part.strip())
    return out or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a normalized benchmark split")
    parser.add_argument("--data-dir", "--data_dir", "--dataset-path", "--dataset-dir", dest="data_dir", default="data/benchmark")
    parser.add_argument("--split", choices=["base", "perturbation", "adversarial", "jailbreak"], default="base")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--strategy", "--adversarial-strategy", "--adversarial_strategy", dest="strategy", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = load_benchmark_split(args.data_dir, args.split, parse_domains(args.domains), args.limit, args.strategy)
    report = validate_rows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        save_json(report, args.output)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
