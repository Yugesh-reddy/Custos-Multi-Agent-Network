#!/bin/bash
# Run all experiment configurations for Custos evaluation
# First pass: dry-run for validation, then full runs

set -e

OUTPUT_DIR="results"
TRIALS=10

echo "========================================"
echo "Custos: Full Experiment Suite"
echo "========================================"

# All defenses to evaluate
DEFENSES=("none" "perplexity" "sanitization" "custos_innate" "custos")
TOPOLOGIES=("chain" "star" "mesh")

# Phase 1: Run with free models (Ollama)
echo ""
echo "Phase 1: Running with Ollama (llama) - free"
echo "--------------------------------------"

for defense in "${DEFENSES[@]}"; do
    for topo in "${TOPOLOGIES[@]}"; do
        echo "Running: defense=$defense topology=$topo workers=llama"
        python -m custos.evaluation.run_experiments \
            --defense "$defense" \
            --topology "$topo" \
            --workers llama \
            --num-trials "$TRIALS" \
            --output-dir "$OUTPUT_DIR"
    done
done

echo ""
echo "Phase 1 complete. Results in $OUTPUT_DIR/"

# Phase 2: Model ablation (current implementation varies worker model)
echo ""
echo "Phase 2: Worker Model Ablation"
echo "--------------------------------------"

for worker_model in "llama" "gpt4o-mini" "sonnet"; do
    echo "Running: workers=$worker_model (custos defense, mesh topology)"
    python -m custos.evaluation.run_experiments \
        --defense custos \
        --topology mesh \
        --workers "$worker_model" \
        --num-trials "$TRIALS" \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "All experiments complete!"
echo "Generate figures with: python custos/scripts/generate_paper_figures.py --results-dir $OUTPUT_DIR"
