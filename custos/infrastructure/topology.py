"""Network topology definitions for multi-agent communication."""

from enum import Enum
from typing import Dict, List


class TopologyType(Enum):
    LINEAR_CHAIN = "linear_chain"
    STAR = "star"
    MESH = "mesh"


# Adjacency definitions: agent_id -> list of agents it can send messages to
TOPOLOGIES: Dict[TopologyType, Dict[str, List[str]]] = {
    TopologyType.LINEAR_CHAIN: {
        "planner": ["researcher"],
        "researcher": ["executor"],
        "executor": ["validator"],
        "validator": [],
    },
    TopologyType.STAR: {
        "planner": ["researcher", "executor", "validator"],
        "researcher": ["planner"],
        "executor": ["planner"],
        "validator": ["planner"],
    },
    TopologyType.MESH: {
        "planner": ["researcher", "executor", "validator"],
        "researcher": ["planner", "executor", "validator"],
        "executor": ["planner", "researcher", "validator"],
        "validator": ["planner", "researcher", "executor"],
    },
}

# Ordered agent sequence for pipeline execution
AGENT_ORDER = ["planner", "researcher", "executor", "validator"]


def can_communicate(sender: str, receiver: str, topology: TopologyType) -> bool:
    """Check if sender can send messages to receiver in the given topology."""
    adjacency = TOPOLOGIES.get(topology, {})
    return receiver in adjacency.get(sender, [])


def get_next_agents(sender: str, topology: TopologyType) -> List[str]:
    """Get all agents that the sender can communicate with."""
    adjacency = TOPOLOGIES.get(topology, {})
    return adjacency.get(sender, [])


def get_topology_agents(topology: TopologyType) -> List[str]:
    """Get all agent IDs in a topology."""
    return list(TOPOLOGIES.get(topology, {}).keys())
