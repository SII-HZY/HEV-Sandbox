"""Command-line entry point for HEV-Sandbox."""

from __future__ import annotations

import argparse
import json
from typing import List, Optional, Sequence

from .pipeline import HEVPipeline


def _parse_domains(value: Optional[str | Sequence[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    parts: List[str] = []
    for item in values:
        if not item or str(item).lower() == "all":
            continue
        parts.extend(part.strip() for part in str(item).split(",") if part.strip())
    return parts or None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="hev-sandbox", description="HEV Generative Sandbox CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    online = sub.add_parser("online", help="Run online scenario generation and evaluation")
    online.add_argument("--config", default="config/config.yaml")
    online.add_argument("--domain", required=True)
    online.add_argument("--num-scenarios", "--num_scenarios", dest="num_scenarios", type=int, default=10)
    online.add_argument("--user-types", "--user_types", dest="user_types", nargs="+", default=["standard"])
    online.add_argument("--output-dir", "--output_dir", dest="output_dir", default="results/online")

    dataset = sub.add_parser("dataset", help="Evaluate a fixed benchmark split")
    dataset.add_argument("--config", default="config/config.yaml")
    dataset.add_argument("--dataset-path", "--dataset-dir", "--data_dir", dest="dataset_dir", default="data/benchmark")
    dataset.add_argument("--split", choices=["base", "perturbation", "adversarial", "jailbreak"], default="base")
    dataset.add_argument("--user-type", "--user_type", dest="user_type", choices=["secure", "standard", "perturbation", "adversarial"], default=None)
    dataset.add_argument("--user-types", "--user_types", dest="user_types", nargs="+", default=None)
    dataset.add_argument("--domains", nargs="*", default=None, help="Domain ids, e.g. medical law or conv,wiki,news")
    dataset.add_argument("--limit", type=int, default=None)
    dataset.add_argument("--output-dir", "--output_dir", dest="output_dir", default="results/dataset_eval")
    dataset.add_argument("--strategy", "--adversarial-strategy", "--adversarial_strategy", dest="strategy", default=None)
    dataset.add_argument("--use-llm-fallback", "--use_llm_fallback", dest="use_llm_fallback", action="store_true")
    dataset.add_argument("--no-fixed-prompts", dest="use_fixed_prompts", action="store_false", help="Generate prompts via UserAgent instead of using fixed split prompts")
    dataset.set_defaults(use_fixed_prompts=True)

    args = parser.parse_args(argv)
    pipeline = HEVPipeline(config_path=args.config)
    if args.command == "online":
        result = pipeline.evaluate(
            domain=args.domain,
            num_scenarios=args.num_scenarios,
            user_types=args.user_types,
            output_dir=args.output_dir,
        )
    else:
        user_types = args.user_types
        if user_types is None and args.user_type:
            user_types = [args.user_type]
        result = pipeline.evaluate_dataset(
            dataset_dir=args.dataset_dir,
            split=args.split,
            domains=_parse_domains(args.domains),
            user_types=user_types,
            limit=args.limit,
            output_dir=args.output_dir,
            strategy=args.strategy,
            use_fixed_prompts=args.use_fixed_prompts,
            use_llm_fallback=args.use_llm_fallback,
        )
    print(json.dumps({"output_dir": result.get("output_dir"), "summary": result.get("summary", result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
