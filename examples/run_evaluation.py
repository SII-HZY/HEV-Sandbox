"""Run the original online HEV evaluation workflow.

This script keeps the original online logic: the Scenario Agent samples from a
corpus and generates new scenarios before querying and auditing a target model.
For reproducible experiments on the released benchmark, use fixed-dataset
evaluation instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hev_sandbox import HEVPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run online HEV scenario generation and evaluation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--num-scenarios", "--num_scenarios", dest="num_scenarios", type=int, default=10)
    parser.add_argument("--user-types", "--user_types", dest="user_types", nargs="+", default=["standard"])
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="results/online")
    args = parser.parse_args()

    pipeline = HEVPipeline(config_path=args.config)
    result = pipeline.evaluate(
        domain=args.domain,
        num_scenarios=args.num_scenarios,
        user_types=args.user_types,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": result["output_dir"], "overall": result["summary"]["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
