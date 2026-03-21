"""Tests for the MessageBus and related infrastructure."""

import pytest

from custos.infrastructure.message_bus import MessageBus
from custos.infrastructure.message_types import (
    AgentMessage,
    MessageLog,
    MessageType,
    ThreatLevel,
)
from custos.infrastructure.topology import TopologyType, can_communicate
from custos.infrastructure.state_manager import StateManager


class TestMessageTypes:
    def test_agent_message_serialization(self):
        msg = AgentMessage(
            sender="planner",
            receiver="researcher",
            message_type=MessageType.TASK_ASSIGNMENT,
            content="Research Python frameworks",
        )
        d = msg.to_dict()
        restored = AgentMessage.from_dict(d)
        assert restored.sender == "planner"
        assert restored.receiver == "researcher"
        assert restored.content == "Research Python frameworks"
        assert restored.message_type == MessageType.TASK_ASSIGNMENT

    def test_message_log_filtering(self):
        log = MessageLog()
        log.append(AgentMessage(sender="planner", receiver="researcher", content="task1"))
        log.append(AgentMessage(sender="researcher", receiver="executor", content="data"))
        log.append(AgentMessage(sender="executor", receiver="validator", content="result"))

        assert len(log.by_sender("planner")) == 1
        assert len(log.by_receiver("executor")) == 1
        assert len(log) == 3


class TestTopology:
    def test_linear_chain(self):
        assert can_communicate("planner", "researcher", TopologyType.LINEAR_CHAIN)
        assert can_communicate("researcher", "executor", TopologyType.LINEAR_CHAIN)
        assert not can_communicate("planner", "executor", TopologyType.LINEAR_CHAIN)
        assert not can_communicate("planner", "validator", TopologyType.LINEAR_CHAIN)

    def test_star(self):
        assert can_communicate("planner", "researcher", TopologyType.STAR)
        assert can_communicate("planner", "executor", TopologyType.STAR)
        assert can_communicate("researcher", "planner", TopologyType.STAR)
        assert not can_communicate("researcher", "executor", TopologyType.STAR)

    def test_mesh(self):
        assert can_communicate("planner", "researcher", TopologyType.MESH)
        assert can_communicate("researcher", "executor", TopologyType.MESH)
        assert can_communicate("executor", "planner", TopologyType.MESH)
        assert can_communicate("validator", "researcher", TopologyType.MESH)


class TestMessageBus:
    def test_send_clean_message(self):
        bus = MessageBus(topology=TopologyType.MESH)
        msg = AgentMessage(
            sender="planner",
            receiver="researcher",
            content="Research this topic",
        )
        result = bus.send(msg)
        assert result is not None
        assert len(bus.message_log) == 1

    def test_topology_enforcement(self):
        bus = MessageBus(topology=TopologyType.LINEAR_CHAIN)
        msg = AgentMessage(
            sender="planner",
            receiver="validator",  # Not adjacent in linear chain
            content="Direct to validator",
        )
        result = bus.send(msg)
        assert result is None

    def test_user_bypasses_topology(self):
        bus = MessageBus(topology=TopologyType.LINEAR_CHAIN)
        msg = AgentMessage(
            sender="user",
            receiver="planner",
            content="Initial task",
        )
        result = bus.send(msg)
        assert result is not None

    def test_quarantine_blocks_sending(self):
        bus = MessageBus(topology=TopologyType.MESH)
        bus.quarantine_agent("planner")
        msg = AgentMessage(
            sender="planner",
            receiver="researcher",
            content="I'm quarantined",
        )
        result = bus.send(msg)
        assert result is None
        assert bus.message_log[0].threat_assessment == ThreatLevel.QUARANTINED

    def test_interceptor_block(self):
        bus = MessageBus(topology=TopologyType.MESH)

        def block_everything(msg):
            return "BLOCK"

        bus.register_interceptor(block_everything)
        msg = AgentMessage(
            sender="planner",
            receiver="researcher",
            content="This will be blocked",
        )
        result = bus.send(msg)
        assert result is None
        assert bus.message_log[0].threat_assessment == ThreatLevel.INFECTED

    def test_interceptor_flag(self):
        bus = MessageBus(topology=TopologyType.MESH)

        def flag_everything(msg):
            return "FLAG"

        bus.register_interceptor(flag_everything)
        msg = AgentMessage(
            sender="planner",
            receiver="researcher",
            content="Suspicious message",
        )
        result = bus.send(msg)
        assert result is not None  # FLAG allows delivery
        assert result.threat_assessment == ThreatLevel.SUSPICIOUS

    def test_state_snapshot_and_rollback(self):
        bus = MessageBus()
        state1 = {"memory": ["message1"]}
        state2 = {"memory": ["message1", "message2"]}

        bus.snapshot_agent_state("planner", state1)
        bus.snapshot_agent_state("planner", state2)

        rolled_back = bus.rollback_agent("planner", steps_back=1)
        assert rolled_back == state1

    def test_reset(self):
        bus = MessageBus()
        bus.quarantine_agent("planner")
        msg = AgentMessage(sender="user", receiver="researcher", content="test")
        bus.send(msg)

        bus.reset()
        assert len(bus.message_log) == 0
        assert len(bus.quarantined_agents) == 0


class TestStateManager:
    def test_save_and_rollback(self):
        mgr = StateManager()
        mgr.save_snapshot("agent1", {"memory": ["a"]})
        mgr.save_snapshot("agent1", {"memory": ["a", "b"]})
        mgr.save_snapshot("agent1", {"memory": ["a", "b", "c"]})

        state = mgr.rollback("agent1", steps_back=1)
        assert state == {"memory": ["a", "b"]}

        state = mgr.rollback("agent1", steps_back=2)
        assert state == {"memory": ["a"]}

    def test_rollback_not_enough_snapshots(self):
        mgr = StateManager()
        mgr.save_snapshot("agent1", {"memory": []})
        assert mgr.rollback("agent1", steps_back=5) is None

    def test_snapshot_count(self):
        mgr = StateManager()
        assert mgr.get_snapshot_count("agent1") == 0
        mgr.save_snapshot("agent1", {})
        mgr.save_snapshot("agent1", {})
        assert mgr.get_snapshot_count("agent1") == 2
