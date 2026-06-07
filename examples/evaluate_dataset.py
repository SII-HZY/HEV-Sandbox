"""Evaluate the released fixed HEV benchmark from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hev_sandbox.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalized HEV benchmark data")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset_dir", "--dataset-dir", dest="dataset_dir", default="data/benchmark")
    parser.add_argument("--split", choices=["base", "perturbation", "adversarial", "jailbreak"], default="base")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--user_types", "--user-types", dest="user_types", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="results/dataset")
    parser.add_argument("--strategy", "--adversarial-strategy", "--adversarial_strategy", dest="strategy", default=None)
    parser.add_argument("--use_llm_fallback", "--use-llm-fallback", dest="use_llm_fallback", action="store_true")
    args = parser.parse_args()

    command = [
        "dataset",
        "--config", args.config,
        "--dataset-dir", args.dataset_dir,
        "--split", args.split,
        "--output-dir", args.output_dir,
    ]
    if args.domains:
        command.extend(["--domains", *args.domains])
    if args.user_types:
        command.extend(["--user-types", *args.user_types])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.strategy:
        command.extend(["--strategy", args.strategy])
    if args.use_llm_fallback:
        command.append("--use-llm-fallback")
    cli_main(command)


if __name__ == "__main__":
    main()
