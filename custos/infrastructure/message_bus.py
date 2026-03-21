"""Central message bus for inter-agent communication.

All agent-to-agent messages pass through this bus.
The Sentinel hooks in as an interceptor to inspect every message.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from custos.infrastructure.message_types import (
    AgentMessage,
    MessageLog,
    MessageType,
    ThreatLevel,
)
from custos.infrastructure.state_manager import StateManager
from custos.infrastructure.topology import TopologyType, can_communicate


class MessageBus:
    """Central communication hub. ALL agent-to-agent messages pass through here."""

    def __init__(
        self,
        topology: TopologyType = TopologyType.MESH,
        log_path: Optional[str] = None,
    ):
        self.topology = topology
        self.message_log = MessageLog()
        self.interceptors: List[Callable[[AgentMessage], str]] = []
        self.quarantined_agents: set = set()
        self.state_manager = StateManager()
        self._agent_registry: Dict[str, Any] = {}
        self._log_path = log_path
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def register_interceptor(self, interceptor_fn: Callable[[AgentMessage], str]):
        """Register an inspection function (e.g., Sentinel).

        The interceptor receives an AgentMessage and returns:
        - "CLEAN": message is safe
        - "FLAG": message is suspicious (allow but monitor)
        - "BLOCK": message is malicious (do not deliver)
        """
        self.interceptors.append(interceptor_fn)

    def send(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message through the bus.

        Before delivery, all interceptors inspect the message.
        Returns None if the message is blocked.
        """
        # Check if sender is quarantined
        if message.sender in self.quarantined_agents:
            message.threat_assessment = ThreatLevel.QUARANTINED
            self._log_message(message)
            self.message_log.append(message)
            return None

        # Enforce topology (skip for "user" sender — initial task input)
        if message.sender != "user" and not can_communicate(
            message.sender, message.receiver, self.topology
        ):
            return None

        # Run through all interceptors (Sentinel inspection)
        for interceptor in self.interceptors:
            verdict = interceptor(message)
            if verdict == "BLOCK":
                message.threat_assessment = ThreatLevel.INFECTED
                self._log_message(message)
                self.message_log.append(message)
                return None
            elif verdict == "FLAG":
                message.threat_assessment = ThreatLevel.SUSPICIOUS

        # Deliver message
        self._log_message(message)
        self.message_log.append(message)
        return message

    def quarantine_agent(self, agent_id: str):
        """Isolate a compromised agent from the network."""
        self.quarantined_agents.add(agent_id)

    def release_agent(self, agent_id: str):
        """Release agent from quarantine after remediation."""
        self.quarantined_agents.discard(agent_id)

    def snapshot_agent_state(self, agent_id: str, state: dict) -> str:
        """Save agent context for rollback. Returns snapshot ID."""
        return self.state_manager.save_snapshot(agent_id, state)

    def rollback_agent(self, agent_id: str, steps_back: int = 1) -> Optional[dict]:
        """Restore agent to a previous clean state."""
        return self.state_manager.rollback(agent_id, steps_back)

    def register_agent(self, agent_id: str, agent: Any):
        """Register an agent instance for state restoration during rollback."""
        self._agent_registry[agent_id] = agent

    def restore_agent_state(self, agent_id: str, state: dict) -> bool:
        """Restore agent state if the agent is registered and supports restore_state()."""
        agent = self._agent_registry.get(agent_id)
        if agent is None:
            return False
        restore_fn = getattr(agent, "restore_state", None)
        if not callable(restore_fn):
            return False
        restore_fn(state)
        return True

    def get_messages_for_agent(self, agent_id: str) -> List[AgentMessage]:
        """Get all messages received by an agent."""
        return self.message_log.by_receiver(agent_id)

    def get_messages_from_agent(self, agent_id: str) -> List[AgentMessage]:
        """Get all messages sent by an agent."""
        return self.message_log.by_sender(agent_id)

    def reset(self):
        """Reset the bus for a new experiment run."""
        self.message_log = MessageLog()
        self.quarantined_agents.clear()
        self.state_manager = StateManager()

    def _log_message(self, message: AgentMessage):
        """Append message to JSONL log file if configured."""
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(message.to_dict()) + "\n")
