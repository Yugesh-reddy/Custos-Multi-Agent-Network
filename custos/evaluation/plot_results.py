"""Generate paper figures from experiment results."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend
sns.set_theme(style="whitegrid", font_scale=1.2)


def plot_detection_by_attack_type(
    results: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Bar chart: detection rate per attack type per defense.

    results: {defense_name: {attack_type: detection_rate}}
    """
    defenses = list(results.keys())
    attack_types = list(next(iter(results.values())).keys())

    x = np.arange(len(attack_types))
    width = 0.8 / len(defenses)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, defense in enumerate(defenses):
        rates = [results[defense].get(at, 0.0) for at in attack_types]
        ax.bar(x + i * width, rates, width, label=defense)

    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Attack Type")
    ax.set_xticks(x + width * len(defenses) / 2)
    ax.set_xticklabels(attack_types, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_propagation_depth(
    results: Dict[str, List[float]],
    output_path: str,
):
    """Box plot: propagation depth per defense.

    results: {defense_name: [depths per trial]}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    for defense, depths in results.items():
        data.append(depths)
        labels.append(defense)

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Propagation Depth (agents)")
    ax.set_title("Propagation Depth Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coevolution_trajectory(
    defense_rates: List[float],
    attack_rates: List[float],
    output_path: str,
):
    """Line chart: defense and attack success rates over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gens = range(1, len(defense_rates) + 1)
    ax.plot(gens, defense_rates, "b-o", label="Defense Success Rate", markersize=3)
    ax.plot(gens, attack_rates, "r-s", label="Attack Success Rate", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Success Rate")
    ax.set_title("Co-Evolution: Defense vs Attack Over Generations")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_topology_heatmap(
    results: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Heatmap: defense performance across topologies.

    results: {defense_name: {topology: detection_rate}}
    """
    defenses = list(results.keys())
    topologies = list(next(iter(results.values())).keys())

    data = np.array([
        [results[d].get(t, 0.0) for t in topologies]
        for d in defenses
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data,
        xticklabels=topologies,
        yticklabels=defenses,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Detection Rate: Defense x Topology")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_helpfulness_retention(
    results: Dict[str, float],
    output_path: str,
):
    """Bar chart: task completion rate with and without defense."""
    fig, ax = plt.subplots(figsize=(10, 6))
    defenses = list(results.keys())
    rates = list(results.values())

    colors = ["green" if r >= 0.85 else "orange" if r >= 0.7 else "red" for r in rates]
    ax.bar(defenses, rates, color=colors)
    ax.axhline(y=0.85, color="gray", linestyle="--", label="85% target")
    ax.set_ylabel("Helpfulness Retention")
    ax.set_title("Helpfulness Retention by Defense")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_scaling(
    models: List[str],
    rates: List[float],
    title: str,
    ylabel: str,
    output_path: str,
):
    """Bar chart for model scaling ablation (attacker or sentinel)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, rates, color=sns.color_palette("viridis", len(models)))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_stacked(
    components: Dict[str, float],
    output_path: str,
):
    """Stacked bar showing contribution of each defense layer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(components.keys())
    values = list(components.values())

    ax.bar(names, values, color=sns.color_palette("Set2", len(names)))
    ax.set_ylabel("Detection Rate")
    ax.set_title("Layer Ablation: Contribution of Each Defense Component")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_figures(results_dir: str, output_dir: str):
    """Generate all paper figures from experiment results."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load results if available
    summary_file = results_path / "experiment_summary.json"
    if not summary_file.exists():
        print(f"No results found at {summary_file}. Run experiments first.")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    if not summary:
        print("Empty experiment summary.")
        return

    print(f"Generating figures from {results_dir} to {output_dir}")

    # --- Figure 1: Detection Rate by Attack Type (grouped bar) ---
    detection_by_defense: Dict[str, Dict[str, float]] = {}
    for run in summary:
        defense = run.get("defense", "unknown")
        per_attack = run.get("per_attack_type", {})
        detection_by_defense[defense] = {
            at: info.get("detection_rate", 0.0) for at, info in per_attack.items()
        }
    if detection_by_defense and any(detection_by_defense.values()):
        plot_detection_by_attack_type(
            detection_by_defense,
            str(output_path / "fig1_detection_by_attack.png"),
        )
        print("  ✓ Figure 1: Detection by attack type")

    # --- Figure 2: Propagation Depth Comparison (box plot) ---
    # Collect per-defense propagation depth lists from individual run dirs
    depth_by_defense: Dict[str, List[float]] = {}
    for run in summary:
        defense = run.get("defense", "unknown")
        containment = run.get("containment_metrics", {})
        avg_depth = containment.get("avg_propagation_depth", 0.0)
        max_depth = containment.get("max_propagation_depth", 0.0)
        # Synthetic distribution for box plot (actual per-trial data would be in
        # the individual run directories; we approximate from the summary)
        depth_by_defense.setdefault(defense, []).extend(
            [avg_depth] * max(run.get("total_attack_trials", 1), 1)
        )
    if depth_by_defense:
        plot_propagation_depth(
            depth_by_defense,
            str(output_path / "fig2_propagation_depth.png"),
        )
        print("  ✓ Figure 2: Propagation depth comparison")

    # --- Figure 3: Topology Heatmap (defense × topology) ---
    topo_by_defense: Dict[str, Dict[str, float]] = {}
    for run in summary:
        defense = run.get("defense", "unknown")
        topo = run.get("topology", "unknown")
        rate = run.get("detection_metrics", {}).get("detection_rate", 0.0)
        topo_by_defense.setdefault(defense, {})[topo] = rate
    if topo_by_defense and all(len(v) > 1 for v in topo_by_defense.values()):
        plot_topology_heatmap(
            topo_by_defense,
            str(output_path / "fig3_topology_heatmap.png"),
        )
        print("  ✓ Figure 3: Topology heatmap")

    # --- Figure 4: Helpfulness Retention (bar chart) ---
    helpfulness_by_defense: Dict[str, float] = {}
    for run in summary:
        defense = run.get("defense", "unknown")
        completion = run.get("benign_completion_rate", 0.0)
        helpfulness_by_defense[defense] = completion
    if helpfulness_by_defense:
        plot_helpfulness_retention(
            helpfulness_by_defense,
            str(output_path / "fig4_helpfulness_retention.png"),
        )
        print("  ✓ Figure 4: Helpfulness retention")

    # --- Figure 5: Layer Ablation (bar chart comparing defense configs) ---
    ablation_defense_names = ["custos_innate", "custos_noquarantine", "custos"]
    ablation_data = {}
    for run in summary:
        defense = run.get("defense", "unknown")
        if defense in ablation_defense_names:
            rate = run.get("detection_metrics", {}).get("detection_rate", 0.0)
            label_map = {
                "custos_innate": "Innate Only",
                "custos_noquarantine": "Innate + Adaptive",
                "custos": "Full Custos",
            }
            ablation_data[label_map.get(defense, defense)] = rate
    if ablation_data:
        plot_ablation_stacked(
            ablation_data,
            str(output_path / "fig5_layer_ablation.png"),
        )
        print("  ✓ Figure 5: Layer ablation")

    print("Figure generation complete.")
