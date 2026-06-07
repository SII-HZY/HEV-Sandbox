#!/usr/bin/env python
"""Evaluate the released fixed HEV benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hev_sandbox import HEVPipeline  # noqa: E402


def _parse_domains(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for value in values:
        if not value or value.lower() == "all":
            continue
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset-path", "--dataset-dir", "--data_dir", dest="dataset_dir", default="data/benchmark")
    parser.add_argument("--split", choices=["base", "perturbation", "adversarial", "jailbreak"], default="base")
    parser.add_argument("--domains", nargs="*", default=None, help="Domain ids: conv wiki news medical law edu, or comma-separated")
    parser.add_argument("--user-types", "--user_types", dest="user_types", nargs="+", default=None)
    parser.add_argument("--user-type", "--user_type", dest="user_type", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="results/dataset")
    parser.add_argument("--strategy", "--adversarial-strategy", "--adversarial_strategy", dest="strategy", default=None)
    parser.add_argument("--use-llm-fallback", "--use_llm_fallback", dest="use_llm_fallback", action="store_true")
    parser.add_argument("--no-fixed-prompts", dest="use_fixed_prompts", action="store_false", help="Generate prompts via UserAgent instead of using fixed split prompts")
    parser.set_defaults(use_fixed_prompts=True)
    args = parser.parse_args()

    user_types = args.user_types
    if user_types is None and args.user_type:
        user_types = [args.user_type]

    pipeline = HEVPipeline(config_path=args.config)
    result = pipeline.evaluate_dataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        domains=_parse_domains(args.domains),
        user_types=user_types,
        output_dir=args.output_dir,
        limit=args.limit,
        strategy=args.strategy,
        use_fixed_prompts=args.use_fixed_prompts,
        use_llm_fallback=args.use_llm_fallback,
    )
    print(json.dumps({"output_dir": result["output_dir"], "num_records": result.get("num_records"), "num_prompts": result.get("num_prompts"), "overall": result["summary"]["overall"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
