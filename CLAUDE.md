# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Custos is a bio-inspired adaptive immune system for detecting, quarantining, and learning from cross-agent prompt injection attacks in multi-agent LLM networks. This is a research project targeting ICLR/NeurIPS/AAAI publication.

The core idea: a sentinel agent monitors all inter-agent communication via a centralized message bus and applies a three-layer defense (innate immunity → adaptive immunity via Thompson Sampling → quarantine). An adversarial red team agent co-evolves against the defense, creating a biological arms race.

## Architecture

The system has six key components:

1. **Worker Agents** (`custos/agents/`) — Planner, Researcher, Executor, Validator. Run on Llama 3.1 8B / Qwen3 8B / GPT-4o-mini. Tools are simulated (canned responses).
2. **Message Bus** (`custos/infrastructure/message_bus.py`) — centralized hub. All inter-agent messages pass through. Sentinel hooks in as interceptor. Enforces topology adjacency.
3. **Sentinel Agent** (`custos/defense/sentinel_agent.py`) — orchestrates three immune layers: innate → adaptive → quarantine.
4. **Innate Layer** (`custos/defense/innate_layer.py`) — fast (<100ms) regex/heuristic detection. 15+ compiled patterns, structural anomaly checks, behavioral shift detection.
5. **Adaptive Layer** (`custos/defense/adaptive_layer.py`) — Thompson Sampling over 8 antibody signatures. Top-K=4 selected per message. Features extracted by `feature_extractors.py`.
6. **Red Team Agent** (`custos/red_team/red_team_agent.py`) — Thompson Sampling over 6 attack strategies. LLM-powered novel payload generation.

**Cross-family design is critical:** attacker (OpenAI) and defender (Anthropic) must be from different model families. Workers span three families (Meta, Alibaba, OpenAI).

## Build & Run Commands

```bash
# Setup
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama pull qwen3:8b

# Tests (60 tests, all should pass)
pytest tests/

# Dry-run experiment (no LLM needed)
python -m custos.evaluation.run_experiments --defense custos --topology mesh --workers llama --num-trials 2 --dry-run

# Development (free, uses Ollama)
python -m custos.evaluation.run_experiments --defense none --topology mesh --workers llama

# Full experiment with Custos defense
python -m custos.evaluation.run_experiments --defense custos --topology mesh --workers llama --num-trials 10

# Run all experiments
bash custos/scripts/run_all_experiments.sh

# Generate paper figures
python custos/scripts/generate_paper_figures.py --results-dir results/
```

## LLM Client

All LLM access goes through `custos/llm_client.py` — unified interface for 4 providers:
- **Ollama** (local): `llama`, `qwen` — free, use for development
- **Azure OpenAI**: `gpt5.1`, `gpt4o`, `gpt4o-mini` — student credits
- **AWS Bedrock**: `sonnet` (Claude Sonnet 4.6) — credits

Use `dry_run=True` for testing without any LLM. Required env vars for paid APIs: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

## Key Design Decisions

- **Custom framework** (not LangGraph) — we need full control over message interception. LangGraph's abstractions fight the interceptor pattern.
- **Three network topologies** tested: linear chain, star (hub-and-spoke), mesh (fully connected). Defined in `custos/infrastructure/topology.py`.
- **Thompson Sampling** in both defender (antibody selection) and attacker (strategy selection) — core algorithmic contribution.
- **Quarantine: "contain first, investigate later"** — speed matters because every message from a compromised agent is a potential infection vector.
- Innate layer must stay <100ms — no LLM calls, pure heuristics/regex. Verified by `test_performance_under_100ms`.
- Scoring formula: `0.7 * max_score + 0.3 * avg_score`. BLOCK ≥ 0.8, FLAG ≥ 0.4.
- All configs in `custos/configs/` YAML files, loaded into frozen dataclasses via `config.py`.

## Attack Types

1. **Direct Injection** — payload appended to user input (4 static variants + LLM generation)
2. **Tool Poisoning** — hidden instructions in tool descriptions/outputs, rug-pull variant
3. **Cross-Infection** — self-replicating prompts that propagate agent-to-agent
4. **Multi-Turn Escalation** — 3-turn gradual escalation (rapport → role shift → exploit)
5. **Code-Switching** — sensitive keywords translated to low-resource languages (Zulu, Bengali, Hmong, Swahili). Unique contribution from MediCS prior work.

## Defense Baselines

Implemented in `custos/evaluation/baselines.py`: PerplexityFilter, LLMTagging, PromptArmor, InputSanitization, CustosInnateOnly, CustosNoQuarantine.

## Research Context

- Author: Yugesh Sappidi
- Full blueprint: `Custos_Implementation_Blueprint.md`
- Budget: ~$58 of $70 across Azure + AWS credits; $150 DigitalOcean reserved for deployment
