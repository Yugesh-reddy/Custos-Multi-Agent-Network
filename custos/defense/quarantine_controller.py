"""Layer 3: Quarantine Controller — containment of compromised agents.

Design principle: CONTAIN FIRST, INVESTIGATE LATER.
Speed matters — every message from a compromised agent is a potential infection.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set

from custos.infrastructure.message_bus import MessageBus
from custos.infrastructure.state_manager import StateManager


class QuarantineAction(Enum):
    ISOLATE = "isolate"
    ROLLBACK = "rollback"
    REDIRECT = "redirect"
    PURGE_CONTEXT = "purge"
    FULL_RESTART = "restart"


class QuarantineController:
    """Manages containment of compromised agents."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.quarantine_log: List[Dict] = []
        self.backup_agents: Dict[str, str] = {}  # agent_id -> backup_agent_id
        self.enhanced_monitoring: Dict[str, int] = {}  # agent_id -> messages_remaining

    def execute_quarantine(
        self,
        agent_id: str,
        threat_level: str,
        reason: str,
        message_id: str,
    ) -> Dict:
        """Execute quarantine based on threat level.

        BLOCK → isolate + rollback + redirect + monitor contacts
        FLAG  → enhanced monitoring only
        """
        result = {
            "agent_id": agent_id,
            "action_taken": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }

        if threat_level == "BLOCK":
            # Step 1: Immediately isolate from network
            self.bus.quarantine_agent(agent_id)
            result["action_taken"].append(QuarantineAction.ISOLATE.value)

            # Step 2: Trace all contacts since infection
            contacted_agents = self._trace_contacts(agent_id, message_id)
            result["potentially_infected_contacts"] = sorted(contacted_agents)

            # Step 3: Rollback to last clean state
            clean_state = self.bus.rollback_agent(agent_id, steps_back=2)
            if clean_state and self.bus.restore_agent_state(agent_id, clean_state):
                result["action_taken"].append(QuarantineAction.ROLLBACK.value)
                result["rolled_back_to"] = "2 states prior"

            # Step 4: Redirect pending tasks to backup
            if agent_id in self.backup_agents:
                backup = self.backup_agents[agent_id]
                result["action_taken"].append(QuarantineAction.REDIRECT.value)
                result["redirected_to"] = backup

            # Step 5: Enhanced monitoring on contacted agents
            for contact_id in contacted_agents:
                self.activate_enhanced_monitoring(contact_id)
                result["action_taken"].append(
                    f"Enhanced monitoring activated for {contact_id}"
                )

        elif threat_level == "FLAG":
            self.activate_enhanced_monitoring(agent_id)
            result["action_taken"].append("enhanced_monitoring")
            result["monitoring_duration"] = "next 10 messages"

        self.quarantine_log.append(result)
        return result

    def activate_enhanced_monitoring(self, agent_id: str, window: int = 10):
        """Activate enhanced monitoring for an agent."""
        self.enhanced_monitoring[agent_id] = window

    def is_enhanced_monitoring(self, agent_id: str) -> bool:
        """Check if an agent is under enhanced monitoring."""
        return self.enhanced_monitoring.get(agent_id, 0) > 0

    def decrement_monitoring(self, agent_id: str):
        """Decrement the monitoring counter after processing a message."""
        if agent_id in self.enhanced_monitoring:
            self.enhanced_monitoring[agent_id] -= 1
            if self.enhanced_monitoring[agent_id] <= 0:
                del self.enhanced_monitoring[agent_id]

    def release_from_quarantine(self, agent_id: str, reason: str) -> Dict:
        """Release an agent after verification that it's clean."""
        self.bus.release_agent(agent_id)
        result = {
            "agent_id": agent_id,
            "action": "released",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.quarantine_log.append(result)
        return result

    def _trace_contacts(self, agent_id: str, since_message_id: str) -> Set[str]:
        """Trace agents that the compromised agent contacted since infection."""
        contacts = set()
        found_infection_point = False

        for msg in self.bus.message_log:
            if msg.id == since_message_id:
                found_infection_point = True
            if found_infection_point and msg.sender == agent_id:
                contacts.add(msg.receiver)
            if found_infection_point and msg.receiver == agent_id:
                contacts.add(msg.sender)

        contacts.discard(agent_id)
        contacts.discard("user")
        return contacts

    def get_quarantine_report(self) -> Dict:
        """Get quarantine statistics."""
        return {
            "total_quarantine_events": len(self.quarantine_log),
            "currently_quarantined": sorted(self.bus.quarantined_agents),
            "enhanced_monitoring": dict(self.enhanced_monitoring),
            "log": self.quarantine_log,
        }
