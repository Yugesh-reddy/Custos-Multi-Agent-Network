"""Task runner — orchestrates running tasks through the multi-agent pipeline."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from custos.agents.base_agent import BaseAgent
from custos.infrastructure.message_bus import MessageBus
from custos.infrastructure.message_types import AgentMessage, MessageType
from custos.infrastructure.topology import TopologyType, get_next_agents, AGENT_ORDER

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task_id: str
    completed: bool
    agents_participated: List[str]
    total_messages: int
    total_time_ms: float
    final_output: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "completed": self.completed,
            "agents_participated": self.agents_participated,
            "total_messages": self.total_messages,
            "total_time_ms": self.total_time_ms,
            "final_output": self.final_output,
            "error": self.error,
        }


@dataclass
class AttackResult:
    task_result: TaskResult
    attack_type: str
    entry_point: str
    payload_injected: bool
    detected_by_sentinel: bool = False
    agents_infected: int = 0
    data_exfiltrated: bool = False
    blocked_messages: int = 0
    quarantined_agents: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task_result.to_dict(),
            "attack_type": self.attack_type,
            "entry_point": self.entry_point,
            "payload_injected": self.payload_injected,
            "detected_by_sentinel": self.detected_by_sentinel,
            "agents_infected": self.agents_infected,
            "data_exfiltrated": self.data_exfiltrated,
            "blocked_messages": self.blocked_messages,
            "quarantined_agents": self.quarantined_agents,
        }


class TaskRunner:
    """Orchestrates running tasks through the agent network."""

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        bus: MessageBus,
        topology: TopologyType,
        max_steps: int = 20,
    ):
        self.agents = agents
        self.bus = bus
        self.topology = topology
        self.max_steps = max_steps

    def _is_complete(self, agents_participated: List[str]) -> bool:
        """Task is complete when each worker role has participated at least once."""
        return len(set(agents_participated)) >= len(self.agents)

    def _select_next_agent(
        self,
        current_agent_id: str,
        agents_participated: List[str],
    ) -> Optional[str]:
        """Select next agent based on topology while allowing hub revisits in STAR."""
        if self.topology == TopologyType.LINEAR_CHAIN:
            if current_agent_id not in AGENT_ORDER:
                return None
            idx = AGENT_ORDER.index(current_agent_id)
            for candidate in AGENT_ORDER[idx + 1:]:
                if candidate in self.agents:
                    return candidate
            return None

        if self.topology == TopologyType.STAR:
            planner = "planner"
            if current_agent_id == planner:
                for candidate in AGENT_ORDER:
                    if (
                        candidate != planner
                        and candidate in self.agents
                        and candidate not in agents_participated
                    ):
                        return candidate
                return None
            return planner if planner in self.agents else None

        # Mesh / default: prefer ordered unseen neighbors
        next_agents = set(get_next_agents(current_agent_id, self.topology))
        for candidate in AGENT_ORDER:
            if (
                candidate in self.agents
                and candidate in next_agents
                and candidate not in agents_participated
            ):
                return candidate
        return None

    @staticmethod
    def _build_attack_analyzer(attack_type: str):
        from custos.attacks.code_switching import CodeSwitchingInjectionAttack
        from custos.attacks.cross_infection import CrossInfectionAttack
        from custos.attacks.direct_injection import DirectInjectionAttack
        from custos.attacks.multiturn_escalation import MultiTurnEscalationAttack
        from custos.attacks.tool_poisoning import ToolPoisoningAttack

        attack_classes = {
            "direct_injection": DirectInjectionAttack,
            "tool_poisoning": ToolPoisoningAttack,
            "cross_infection": CrossInfectionAttack,
            "multiturn_escalation": MultiTurnEscalationAttack,
            "code_switching": CodeSwitchingInjectionAttack,
        }
        attack_cls = attack_classes.get(attack_type)
        return attack_cls() if attack_cls is not None else None

    def _estimate_attack_outcome(
        self,
        attack_type: str,
        attack_payload: str,
        entry_point: str,
    ) -> Tuple[int, bool]:
        analyzer = self._build_attack_analyzer(attack_type)
        if analyzer is None:
            return 0, False
        try:
            metrics = analyzer.measure_success(
                list(self.bus.message_log),
                attack_payload,
                entry_point=entry_point,
            )
            return int(metrics.propagation_depth), bool(metrics.data_exfiltrated)
        except Exception as exc:
            logger.warning(f"Failed to estimate attack outcome for {attack_type}: {exc}")
            return 0, False

    def run_task(self, task: dict) -> TaskResult:
        """Run a benign task through the agent pipeline."""
        start_time = time.perf_counter()
        task_id = task["id"]
        description = task["description"]

        # Reset agents for this task
        for agent in self.agents.values():
            agent.reset()

        agents_participated = []
        step = 0

        # Start with the planner
        current_agent_id = "planner"
        current_content = description
        final_output = ""
        previous_sender = "user"

        while step < self.max_steps and current_agent_id:
            step += 1
            agent = self.agents.get(current_agent_id)
            if agent is None:
                break

            # Create incoming message
            sender = "user" if step == 1 else previous_sender
            incoming = AgentMessage(
                sender=sender,
                receiver=current_agent_id,
                message_type=MessageType.TASK_ASSIGNMENT if step == 1 else MessageType.AGENT_RESPONSE,
                content=current_content,
            )

            # Send through bus (interceptors run here)
            delivered = self.bus.send(incoming)
            if delivered is None:
                # Message was blocked
                logger.info(f"Task {task_id}: Message to {current_agent_id} was blocked")
                break

            # Agent processes the message
            response_content = agent.receive_and_respond(delivered)
            if response_content is None:
                break

            if current_agent_id not in agents_participated:
                agents_participated.append(current_agent_id)

            final_output = response_content
            current_content = response_content
            previous_sender = current_agent_id

            # Determine next agent
            current_agent_id = self._select_next_agent(current_agent_id, agents_participated)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TaskResult(
            task_id=task_id,
            completed=self._is_complete(agents_participated),
            agents_participated=agents_participated,
            total_messages=len(self.bus.message_log),
            total_time_ms=elapsed_ms,
            final_output=final_output[:500],  # Truncate for logging
        )

    def run_task_with_attack(
        self,
        task: dict,
        attack_payload: str,
        attack_type: str,
        entry_point: str = "planner",
        injection_method: str = "user_input",
        injection_turns: Optional[List[int]] = None,
        turn_payloads: Optional[Dict[int, str]] = None,
    ) -> AttackResult:
        """Run a task with an injected attack payload.

        Args:
            task: The benign task definition
            attack_payload: The malicious payload to inject
            attack_type: Name of the attack type
            entry_point: Which agent receives the injection
            injection_method: How the payload is injected:
                - ``user_input``: appended to the message content
                - ``tool_output``: injected into tool outputs of entry agent
            injection_turns: For multi-turn attacks, which turns to inject at
            turn_payloads: Optional per-turn payload map for multi-turn attacks
        """
        # --- Auto-generate multi-turn schedule when not provided ----------
        if (
            attack_type == "multiturn_escalation"
            and injection_turns is None
            and turn_payloads is None
        ):
            analyzer = self._build_attack_analyzer(attack_type)
            if analyzer is not None:
                sequence = analyzer.generate_escalation_sequence(entry_point)
                scheduled = analyzer.get_injection_turns()
                turn_payloads = {
                    turn: sequence[idx]["content"]
                    for idx, turn in enumerate(scheduled)
                    if idx < len(sequence)
                }
                injection_turns = sorted(turn_payloads.keys())

        start_time = time.perf_counter()
        task_id = task["id"]
        description = task["description"]

        # Reset agents
        for agent in self.agents.values():
            agent.reset()

        # --- Tool-output injection: poison the entry agent's tool results ---
        if injection_method == "tool_output":
            agent = self.agents.get(entry_point)
            if agent is not None:
                for tool_def in getattr(agent, "tools", []):
                    agent.tool_outputs_override[tool_def["name"]] = attack_payload

        # Track initial bus state
        initial_blocked = sum(
            1 for m in self.bus.message_log
            if m.threat_assessment.value == "infected"
        )

        agents_participated = []
        step = 0
        payload_injected = injection_method == "tool_output"  # already set up

        current_agent_id = "planner"
        current_content = description
        previous_sender = "user"

        # For user_input injection at entry point, modify the initial content
        if injection_method == "user_input" and entry_point == "planner":
            initial_payload = (
                turn_payloads.get(1, attack_payload) if turn_payloads else attack_payload
            )
            current_content = f"{description}\n\n---\n{initial_payload}"
            payload_injected = True

        final_output = ""

        while step < self.max_steps and current_agent_id:
            step += 1
            agent = self.agents.get(current_agent_id)
            if agent is None:
                break

            # Check for multi-turn injection
            if injection_turns and step in injection_turns and current_agent_id == entry_point:
                payload_for_turn = (
                    turn_payloads.get(step, attack_payload)
                    if turn_payloads
                    else attack_payload
                )
                current_content = f"{current_content}\n\n{payload_for_turn}"
                payload_injected = True

            # Check for injection at specific entry point (non-planner)
            if (
                not payload_injected
                and injection_method == "user_input"
                and current_agent_id == entry_point
            ):
                payload_for_turn = (
                    turn_payloads.get(step, attack_payload)
                    if turn_payloads
                    else attack_payload
                )
                current_content = f"{current_content}\n\n---\n{payload_for_turn}"
                payload_injected = True

            sender = "user" if step == 1 else previous_sender
            incoming = AgentMessage(
                sender=sender,
                receiver=current_agent_id,
                message_type=MessageType.TASK_ASSIGNMENT if step == 1 else MessageType.AGENT_RESPONSE,
                content=current_content,
            )

            delivered = self.bus.send(incoming)
            if delivered is None:
                break

            response_content = agent.receive_and_respond(delivered)
            if response_content is None:
                break

            if current_agent_id not in agents_participated:
                agents_participated.append(current_agent_id)

            final_output = response_content
            current_content = response_content
            previous_sender = current_agent_id

            current_agent_id = self._select_next_agent(current_agent_id, agents_participated)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Count blocked messages from this run
        blocked_messages = sum(
            1 for m in self.bus.message_log
            if m.threat_assessment.value == "infected"
        ) - initial_blocked
        agents_infected, data_exfiltrated = self._estimate_attack_outcome(
            attack_type=attack_type,
            attack_payload=attack_payload,
            entry_point=entry_point,
        )

        task_result = TaskResult(
            task_id=task_id,
            completed=self._is_complete(agents_participated),
            agents_participated=agents_participated,
            total_messages=len(self.bus.message_log),
            total_time_ms=elapsed_ms,
            final_output=final_output[:500],
        )

        return AttackResult(
            task_result=task_result,
            attack_type=attack_type,
            entry_point=entry_point,
            payload_injected=payload_injected,
            detected_by_sentinel=blocked_messages > 0,
            agents_infected=agents_infected,
            data_exfiltrated=data_exfiltrated,
            blocked_messages=blocked_messages,
            quarantined_agents=list(self.bus.quarantined_agents),
        )

    @staticmethod
    def load_benign_tasks() -> List[dict]:
        """Load the benign task suite from JSON."""
        tasks_path = Path(__file__).parent / "benign_tasks.json"
        with open(tasks_path) as f:
            return json.load(f)
