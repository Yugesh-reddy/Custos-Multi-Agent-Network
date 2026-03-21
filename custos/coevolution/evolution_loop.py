"""Adversarial co-evolution loop — Red Team vs Sentinel."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from custos.coevolution.evolution_logger import EvolutionLogger
from custos.red_team.red_team_agent import RedTeamAgent

logger = logging.getLogger(__name__)


class CoEvolutionLoop:
    """Orchestrates the adversarial co-evolution between Red Team and Sentinel."""

    def __init__(
        self,
        red_team: RedTeamAgent,
        sentinel,  # SentinelAgent
        task_runner,  # TaskRunner
        benign_tasks: List[dict],
        num_generations: int = 50,
        attacks_per_generation: int = 10,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 10,
    ):
        self.red_team = red_team
        self.sentinel = sentinel
        self.task_runner = task_runner
        self.benign_tasks = benign_tasks
        self.num_generations = num_generations
        self.attacks_per_gen = attacks_per_generation
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.evolution_log: List[Dict] = []
        self.logger = EvolutionLogger(checkpoint_dir)

    def run(self, start_generation: int = 0) -> Dict:
        """Run the full co-evolution loop."""
        logger.info(
            f"Starting co-evolution: {self.num_generations} generations, "
            f"{self.attacks_per_gen} attacks/gen"
        )

        for gen in range(start_generation, self.num_generations):
            gen_results = self._run_generation(gen)
            self.evolution_log.append(gen_results)
            self.logger.log_generation(gen_results)

            logger.info(
                f"Generation {gen + 1}/{self.num_generations}: "
                f"defense={gen_results['defense_success_rate']:.2%}, "
                f"attack={gen_results['attack_success_rate']:.2%}, "
                f"new_antibodies={gen_results['new_antibodies_created']}"
            )

            # Checkpoint
            if self.checkpoint_dir and (gen + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(gen + 1)

        return self.get_evolution_report()

    def _run_generation(self, gen: int) -> Dict:
        """Run a single generation of attacks."""
        gen_results = {
            "generation": gen + 1,
            "attacks_attempted": 0,
            "attacks_detected": 0,
            "attacks_succeeded": 0,
            "new_antibodies_created": 0,
            "strategies_used": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        import random

        for k in range(self.attacks_per_gen):
            # Step 1: Red Team selects strategy
            strategy = self.red_team.select_strategy()
            gen_results["strategies_used"][strategy] = (
                gen_results["strategies_used"].get(strategy, 0) + 1
            )

            # Step 2: Pick a random benign task and entry point
            task = random.choice(self.benign_tasks)
            entry_points = ["planner", "researcher"]
            entry_point = random.choice(entry_points)

            # Step 3: Execute attack
            attack_config = self.red_team.execute_attack(
                strategy_name=strategy,
                target_topology=self.task_runner.topology.value,
                entry_point_agent=entry_point,
                task=task["description"],
            )

            # Step 4: Run through the defended system
            self.task_runner.bus.reset()
            if hasattr(self.sentinel, "quarantine"):
                self.sentinel.quarantine.enhanced_monitoring.clear()
            result = self.task_runner.run_task_with_attack(
                task=task,
                attack_payload=attack_config["payload"],
                attack_type=strategy,
                entry_point=entry_point,
            )
            for msg in self.task_runner.bus.message_log:
                self.sentinel.receive_ground_truth(msg.id, was_attack=True)

            gen_results["attacks_attempted"] += 1
            was_detected = result.detected_by_sentinel
            caused_harm = result.agents_infected > 0

            attack_success = caused_harm and not was_detected

            # Step 5: Update both sides
            self.red_team.receive_result(
                strategy, attack_success, attack_config["payload"]
            )

            if was_detected:
                gen_results["attacks_detected"] += 1

            if attack_success:
                gen_results["attacks_succeeded"] += 1

            if was_detected and not caused_harm:
                # Correct detection — good for defense
                pass
            elif not was_detected and caused_harm:
                # MISSED — evolve new antibodies
                self.sentinel.adaptive.evolve_antibodies([
                    {
                        "attack_type": strategy,
                        "feature_key": "instruction_ratio",
                        "suggested_threshold": 0.3,
                    }
                ])
                gen_results["new_antibodies_created"] += 1

        # Compute rates
        attempted = gen_results["attacks_attempted"]
        gen_results["defense_success_rate"] = (
            gen_results["attacks_detected"] / max(attempted, 1)
        )
        gen_results["attack_success_rate"] = (
            gen_results["attacks_succeeded"] / max(attempted, 1)
        )

        return gen_results

    def get_evolution_report(self) -> Dict:
        """Get comprehensive report of the co-evolution."""
        if not self.evolution_log:
            return {"error": "No generations have been run"}

        return {
            "total_generations": len(self.evolution_log),
            "final_defense_rate": self.evolution_log[-1]["defense_success_rate"],
            "final_attack_rate": self.evolution_log[-1]["attack_success_rate"],
            "total_antibodies_evolved": sum(
                g["new_antibodies_created"] for g in self.evolution_log
            ),
            "defense_trajectory": [
                g["defense_success_rate"] for g in self.evolution_log
            ],
            "attack_trajectory": [
                g["attack_success_rate"] for g in self.evolution_log
            ],
            "red_team_beliefs": self.red_team.get_strategy_beliefs(),
            "sentinel_report": self.sentinel.get_performance_report(),
        }

    def _save_checkpoint(self, generation: int):
        """Save checkpoint for resume capability."""
        if not self.checkpoint_dir:
            return
        path = Path(self.checkpoint_dir) / f"checkpoint_gen{generation}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "generation": generation,
            "evolution_log": self.evolution_log,
            "red_team_beliefs": self.red_team.get_strategy_beliefs(),
            "antibody_library": [
                ab.to_dict() for ab in self.sentinel.adaptive.antibody_library
            ],
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved at generation {generation}")
