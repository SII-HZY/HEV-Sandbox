"""Example: evaluate the fixed HEV-Sandbox benchmark."""

from hev_sandbox import HEVPipeline

pipeline = HEVPipeline(config_path="config/config.yaml")

result = pipeline.evaluate_dataset(
    dataset_path="data/benchmark",
    split="base",
    domains=["medical"],
    limit=5,
    user_types=["standard"],
    output_dir="results/example_dataset",
)

print("Output directory:", result["output_dir"])
print("Overall:", result["summary"]["overall"])
