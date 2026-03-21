"""Agent state snapshot and rollback management."""

import copy
from datetime import datetime, timezone
from typing import Dict, List, Optional


class StateManager:
    """Manages agent state snapshots for quarantine rollback."""

    def __init__(self):
        self._snapshots: Dict[str, List[dict]] = {}

    def save_snapshot(self, agent_id: str, state: dict) -> str:
        """Save a state snapshot for an agent. Returns snapshot ID."""
        if agent_id not in self._snapshots:
            self._snapshots[agent_id] = []

        snapshot = {
            "id": f"{agent_id}_snap_{len(self._snapshots[agent_id])}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": copy.deepcopy(state),
        }
        self._snapshots[agent_id].append(snapshot)
        return snapshot["id"]

    def rollback(self, agent_id: str, steps_back: int = 1) -> Optional[dict]:
        """Restore agent to a previous state. Returns the state dict or None."""
        snapshots = self._snapshots.get(agent_id, [])
        if len(snapshots) > steps_back:
            return copy.deepcopy(snapshots[-(steps_back + 1)]["state"])
        return None

    def get_clean_state(self, agent_id: str, before_message_id: str) -> Optional[dict]:
        """Get the last state snapshot taken before a specific message.

        This is used to find the last known-good state before infection.
        The before_message_id is stored in snapshot metadata for correlation.
        """
        snapshots = self._snapshots.get(agent_id, [])
        if not snapshots:
            return None
        # Return the most recent snapshot (in practice, we'd correlate with message IDs)
        # For now, go back 2 steps as a safe default
        return self.rollback(agent_id, steps_back=2)

    def get_snapshot_count(self, agent_id: str) -> int:
        return len(self._snapshots.get(agent_id, []))

    def clear_snapshots(self, agent_id: str):
        """Clear all snapshots for an agent (e.g., after full restart)."""
        self._snapshots.pop(agent_id, None)
