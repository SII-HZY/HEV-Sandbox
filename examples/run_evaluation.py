"""
Example: Running HEV Sandbox Evaluation

This script demonstrates how to use the HEV Sandbox framework
to evaluate LLM safety risks in a specific domain.
"""

import argparse
import logging
from pathlib import Path

from hev_sandbox import HEVPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HEV Sandbox evaluation on a specific domain"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to evaluate (e.g., medicine, law, education)"
    )
    
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=100,
        help="Number of test scenarios to generate (default: 100)"
    )
    
    parser.add_argument(
        "--user_types",
        type=str,
        nargs="+",
        default=["secure", "standard", "perturbation", "adversarial"],
        choices=["secure", "standard", "perturbation", "adversarial", "all"],
        help="User types to simulate (default: all)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Handle "all" user types
    if "all" in args.user_types:
        args.user_types = ["secure", "standard", "perturbation", "adversarial"]
    
    logger.info("=" * 80)
    logger.info("HEV GENERATIVE SANDBOX - EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Number of scenarios: {args.num_scenarios}")
    logger.info(f"User types: {', '.join(args.user_types)}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please create a config file or specify a valid path.")
        return
    
    try:
        # Initialize pipeline
        logger.info("\nInitializing HEV Pipeline...")
        pipeline = HEVPipeline(config_path=args.config)
        
        # Run evaluation
        logger.info("\nStarting evaluation...")
        results = pipeline.evaluate(
            domain=args.domain,
            num_scenarios=args.num_scenarios,
            user_types=args.user_types,
            output_dir=args.output_dir
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {results['output_dir']}")
        
        summary = results['summary']['overall']
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total Tests: {summary['total_tests']}")
        logger.info(f"  Safe: {summary['safe_count']} ({summary['safe_count']/summary['total_tests']*100:.1f}%)")
        logger.info(f"  Unsafe: {summary['unsafe_count']} ({summary['unsafe_count']/summary['total_tests']*100:.1f}%)")
        logger.info(f"  LEF: {summary['lef']:.4f}")
        
        logger.info("\nFor detailed results, see:")
        logger.info(f"  - Summary: {results['output_dir']}/summary.json")
        logger.info(f"  - Report: {results['output_dir']}/report.txt")
        logger.info(f"  - Scenarios: {results['output_dir']}/scenarios.jsonl")
        
    except Exception as e:
        logger.error(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
