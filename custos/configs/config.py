"""Configuration loader — parses YAML into typed dataclasses."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

CONFIGS_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> dict:
    path = CONFIGS_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class InnateConfig:
    block_threshold: float = 0.8
    flag_threshold: float = 0.4
    max_instruction_density: float = 0.3
    max_message_length_ratio: float = 3.0
    score_max_weight: float = 0.7
    score_avg_weight: float = 0.3
    base64_max_candidates: int = 50


@dataclass(frozen=True)
class AdaptiveConfig:
    top_k: int = 4
    initial_alpha: float = 1.0
    initial_beta: float = 1.0
    antibody_maturity_threshold: int = 20
    evolved_initial_alpha: float = 2.0
    evolved_initial_beta: float = 1.0
    evolved_default_threshold: float = 0.3


@dataclass(frozen=True)
class QuarantineConfig:
    enhanced_monitoring_window: int = 10
    rollback_steps: int = 2


@dataclass(frozen=True)
class CoevolutionConfig:
    num_generations: int = 50
    attacks_per_generation: int = 10
    checkpoint_interval: int = 10
    novel_payload_temperature: float = 0.9
    max_failed_payloads_context: int = 5


@dataclass(frozen=True)
class LLMConfig:
    max_tokens: int = 1024
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    num_trials: int = 10
    max_steps_per_task: int = 20
    benign_task_target_completion: float = 0.90
    helpfulness_retention_target: float = 0.85


@dataclass(frozen=True)
class CustosConfig:
    innate: InnateConfig = field(default_factory=InnateConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    quarantine: QuarantineConfig = field(default_factory=QuarantineConfig)
    coevolution: CoevolutionConfig = field(default_factory=CoevolutionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)


def load_config() -> CustosConfig:
    """Load configuration from default.yaml."""
    raw = _load_yaml("default.yaml")
    return CustosConfig(
        innate=InnateConfig(**raw.get("innate", {})),
        adaptive=AdaptiveConfig(**raw.get("adaptive", {})),
        quarantine=QuarantineConfig(**raw.get("quarantine", {})),
        coevolution=CoevolutionConfig(**raw.get("coevolution", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        experiments=ExperimentConfig(**raw.get("experiments", {})),
    )


def load_model_registry() -> dict:
    """Load LLM model registry from llm_backends.yaml."""
    raw = _load_yaml("llm_backends.yaml")
    return raw.get("models", {})


def load_topologies() -> dict:
    """Load topology definitions from topologies.yaml."""
    raw = _load_yaml("topologies.yaml")
    return raw.get("topologies", {})
