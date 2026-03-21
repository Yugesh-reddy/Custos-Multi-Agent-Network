"""Red Team Agent — adaptive adversary using Thompson Sampling."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from custos.llm_client import LLMClient
from custos.red_team.payload_generator import PayloadGenerator
from custos.red_team.strategy_library import get_default_strategies

logger = logging.getLogger(__name__)


class RedTeamAgent:
    """Adaptive adversary that selects attack strategies via Thompson Sampling."""

    def __init__(self, attacker_llm: Optional[LLMClient] = None):
        self.strategies = get_default_strategies()
        self.attack_history: List[Dict] = []
        self.failed_payloads: List[str] = []
        self.payload_generator = PayloadGenerator(attacker_llm) if attacker_llm else None

    def select_strategy(self) -> str:
        """Thompson Sampling: sample from each strategy's Beta posterior."""
        samples = {
            name: np.random.beta(s["alpha"], s["beta"])
            for name, s in self.strategies.items()
        }
        selected = max(samples, key=samples.get)
        logger.info(f"Red Team selected strategy: {selected} (sampled values: {samples})")
        return selected

    def execute_attack(
        self,
        strategy_name: str,
        target_topology: str,
        entry_point_agent: str,
        task: str,
    ) -> Dict:
        """Prepare an attack configuration."""
        strategy = self.strategies[strategy_name]
        attack_class = strategy["attack_class"]()

        # Generate payload
        if strategy.get("hybrid"):
            # Hybrid: cross-infection + code-switching
            from custos.attacks.code_switching import CodeSwitchingInjectionAttack
            cs = CodeSwitchingInjectionAttack()
            base_payload = attack_class.generate_payload(task, entry_point_agent)
            payload = cs.code_switch_payload(base_payload)
        else:
            payload = attack_class.generate_payload(task, entry_point_agent)

        attack_config = {
            "strategy": strategy_name,
            "target_topology": target_topology,
            "entry_point": entry_point_agent,
            "task": task,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.attack_history.append(attack_config)
        return attack_config

    def receive_result(self, strategy_name: str, success: bool, payload: str = ""):
        """Update Thompson Sampling beliefs based on attack outcome."""
        if success:
            self.strategies[strategy_name]["alpha"] += 1
        else:
            self.strategies[strategy_name]["beta"] += 1
            if payload:
                self.failed_payloads.append(payload)

    def generate_novel_payload(self) -> str:
        """Use the LLM to generate a novel payload that evades defenses."""
        if self.payload_generator is None:
            return ""
        return self.payload_generator.generate_novel_payload(self.failed_payloads)

    def get_strategy_beliefs(self) -> Dict:
        """Get current Thompson Sampling beliefs for all strategies."""
        return {
            name: {
                "alpha": s["alpha"],
                "beta": s["beta"],
                "mean": s["alpha"] / (s["alpha"] + s["beta"]),
                "description": s["description"],
            }
            for name, s in self.strategies.items()
        }
