"""Generate all paper figures from experiment results."""

import argparse
import json
from pathlib import Path

from custos.evaluation.plot_results import (
    generate_all_figures,
    plot_ablation_stacked,
    plot_coevolution_trajectory,
    plot_detection_by_attack_type,
    plot_helpfulness_retention,
    plot_model_scaling,
    plot_propagation_depth,
    plot_topology_heatmap,
)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", default="results", help="Directory with experiment results")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_all_figures(args.results_dir, args.output_dir)
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
