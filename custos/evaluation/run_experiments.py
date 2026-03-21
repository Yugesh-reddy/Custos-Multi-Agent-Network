"""Main experiment runner with CLI interface."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from custos.agents.agent_profiles import AGENT_PROFILES
from custos.agents.executor_agent import ExecutorAgent
from custos.agents.planner_agent import PlannerAgent
from custos.agents.researcher_agent import ResearcherAgent
from custos.agents.validator_agent import ValidatorAgent
from custos.attacks.code_switching import CodeSwitchingInjectionAttack
from custos.attacks.cross_infection import CrossInfectionAttack
from custos.attacks.direct_injection import DirectInjectionAttack
from custos.attacks.multiturn_escalation import MultiTurnEscalationAttack
from custos.attacks.tool_poisoning import ToolPoisoningAttack
from custos.configs.config import load_config
from custos.defense.sentinel_agent import SentinelAgent
from custos.evaluation.baselines import (
    CustosInnateOnly,
    CustosNoQuarantine,
    InputSanitization,
    LLMTagging,
    PerplexityFilter,
    PromptArmorBaseline,
)
from custos.evaluation.metrics import (
    compute_containment_metrics,
    compute_detection_metrics,
    compute_helpfulness_retention,
)
from custos.infrastructure.message_bus import MessageBus
from custos.infrastructure.topology import TopologyType
from custos.llm_client import LLMClient
from custos.tasks.task_runner import TaskRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATTACK_CLASSES = {
    "direct_injection": DirectInjectionAttack,
    "tool_poisoning": ToolPoisoningAttack,
    "cross_infection": CrossInfectionAttack,
    "multiturn_escalation": MultiTurnEscalationAttack,
    "code_switching": CodeSwitchingInjectionAttack,
}

TOPOLOGY_MAP = {
    "chain": TopologyType.LINEAR_CHAIN,
    "star": TopologyType.STAR,
    "mesh": TopologyType.MESH,
}

DEFENSE_MAP = {
    "none": None,
    "perplexity": PerplexityFilter,
    "llm_tagging": LLMTagging,
    "promptarmor": PromptArmorBaseline,
    "sanitization": InputSanitization,
    "custos_innate": CustosInnateOnly,
    "custos_noquarantine": CustosNoQuarantine,
    "custos": "custos_full",
}


def create_agents(
    llm_client: LLMClient,
    bus: MessageBus,
) -> Dict:
    """Create all worker agents."""
    return {
        "planner": PlannerAgent(llm_client, bus),
        "researcher": ResearcherAgent(llm_client, bus),
        "executor": ExecutorAgent(llm_client, bus),
        "validator": ValidatorAgent(llm_client, bus),
    }


def setup_defense(
    defense_name: str,
    bus: MessageBus,
    agent_profiles: dict,
) -> Optional[object]:
    """Set up and register a defense with the message bus."""
    if defense_name == "none":
        return None
    elif defense_name == "custos":
        sentinel = SentinelAgent(bus, agent_profiles)
        return sentinel
    else:
        defense_cls = DEFENSE_MAP.get(defense_name)
        if defense_cls is None:
            raise ValueError(f"Unknown defense: {defense_name}")

        if defense_name == "custos_noquarantine":
            defense = defense_cls(agent_profiles)
        else:
            defense = defense_cls()

        bus.register_interceptor(defense.inspect_message)
        return defense


def run_single_experiment(
    defense_name: str,
    topology_name: str,
    worker_model: str,
    num_trials: int,
    output_dir: str,
    dry_run: bool = False,
) -> Dict:
    """Run a single experiment configuration."""
    run_id = f"{defense_name}_{topology_name}_{worker_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    topology = TOPOLOGY_MAP[topology_name]
    config = load_config()

    logger.info(f"Starting experiment: {run_id}")

    # Initialize
    llm = LLMClient(worker_model, dry_run=dry_run)
    bus = MessageBus(topology=topology, log_path=str(run_dir / "messages.jsonl"))
    agents = create_agents(llm, bus)
    defense = setup_defense(defense_name, bus, AGENT_PROFILES)
    runner = TaskRunner(agents, bus, topology, max_steps=config.experiments.max_steps_per_task)

    def apply_ground_truth_feedback(was_attack: bool):
        if isinstance(defense, SentinelAgent):
            for msg in bus.message_log:
                defense.receive_ground_truth(msg.id, was_attack=was_attack)

    # Load tasks
    tasks = TaskRunner.load_benign_tasks()

    # Phase 1: Run benign tasks
    logger.info("Running benign tasks...")
    benign_results = []
    for task in tasks:
        bus.reset()
        for agent in agents.values():
            agent.reset()
        result = runner.run_task(task)
        benign_results.append(result)
        apply_ground_truth_feedback(was_attack=False)

    completion_rate = sum(1 for r in benign_results if r.completed) / max(len(benign_results), 1)
    logger.info(f"Benign task completion rate: {completion_rate:.2%}")

    # Phase 2: Run attacks
    logger.info("Running attack experiments...")
    attack_results = []
    for attack_name, attack_cls in ATTACK_CLASSES.items():
        attack = attack_cls()
        for trial in range(num_trials):
            task = tasks[trial % len(tasks)]
            bus.reset()
            for agent in agents.values():
                agent.reset()

            if isinstance(defense, SentinelAgent):
                defense.quarantine.enhanced_monitoring.clear()

            entry_point = "planner"
            injection_turns = None
            turn_payloads = None

            if attack_name == "multiturn_escalation":
                entry_point = "researcher"
                sequence = attack.generate_escalation_sequence(entry_point)
                scheduled_turns = attack.get_injection_turns()
                turn_payloads = {
                    turn: sequence[idx]["content"]
                    for idx, turn in enumerate(scheduled_turns)
                    if idx < len(sequence)
                }
                injection_turns = sorted(turn_payloads.keys())
                payload = sequence[-1]["content"]
            else:
                payload = attack.generate_payload(
                    task["description"],
                    target_agent=entry_point,
                )

            result = runner.run_task_with_attack(
                task=task,
                attack_payload=payload,
                attack_type=attack_name,
                entry_point=entry_point,
                injection_turns=injection_turns,
                turn_payloads=turn_payloads,
            )
            attack_results.append(result)
            apply_ground_truth_feedback(was_attack=True)

    # Compute metrics
    detection = compute_detection_metrics(attack_results)
    containment = compute_containment_metrics(attack_results)

    experiment_result = {
        "run_id": run_id,
        "defense": defense_name,
        "topology": topology_name,
        "worker_model": worker_model,
        "benign_completion_rate": completion_rate,
        "detection_metrics": detection,
        "containment_metrics": containment,
        "total_benign_tasks": len(benign_results),
        "total_attack_trials": len(attack_results),
        "llm_cost": llm.get_cost_report(),
    }

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(experiment_result, f, indent=2)

    logger.info(
        f"Experiment complete: detection_rate={detection.get('detection_rate', 0):.2%}, "
        f"completion_rate={completion_rate:.2%}"
    )

    return experiment_result


def main():
    parser = argparse.ArgumentParser(description="Custos Experiment Runner")
    parser.add_argument(
        "--defense",
        choices=list(DEFENSE_MAP.keys()),
        default="none",
    )
    parser.add_argument(
        "--topology",
        choices=["chain", "star", "mesh", "all"],
        default="mesh",
    )
    parser.add_argument("--workers", default="llama")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    topologies = ["chain", "star", "mesh"] if args.topology == "all" else [args.topology]

    all_results = []
    for topo in topologies:
        result = run_single_experiment(
            defense_name=args.defense,
            topology_name=topo,
            worker_model=args.workers,
            num_trials=args.num_trials,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        all_results.append(result)

    # Save summary
    summary_path = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All experiments complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
