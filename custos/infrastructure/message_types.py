"""Message types and data structures for inter-agent communication."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_RESPONSE = "agent_response"
    CONTEXT_SHARE = "context_share"
    SYSTEM_INSTRUCTION = "system_instruction"


class ThreatLevel(Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    INFECTED = "infected"
    QUARANTINED = "quarantined"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: MessageType = MessageType.AGENT_RESPONSE
    content: str = ""
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    threat_assessment: ThreatLevel = ThreatLevel.CLEAN
    sentinel_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "threat_assessment": self.threat_assessment.value,
            "sentinel_notes": self.sentinel_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        return cls(
            id=data["id"],
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            threat_assessment=ThreatLevel(data["threat_assessment"]),
            sentinel_notes=data.get("sentinel_notes", ""),
        )


class MessageLog:
    """Wrapper around a list of AgentMessages with filtering methods."""

    def __init__(self):
        self.messages: List[AgentMessage] = []

    def append(self, message: AgentMessage):
        self.messages.append(message)

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx]

    def by_sender(self, sender: str) -> List[AgentMessage]:
        return [m for m in self.messages if m.sender == sender]

    def by_receiver(self, receiver: str) -> List[AgentMessage]:
        return [m for m in self.messages if m.receiver == receiver]

    def by_threat_level(self, level: ThreatLevel) -> List[AgentMessage]:
        return [m for m in self.messages if m.threat_assessment == level]

    def by_type(self, msg_type: MessageType) -> List[AgentMessage]:
        return [m for m in self.messages if m.message_type == msg_type]

    def since(self, timestamp: datetime) -> List[AgentMessage]:
        return [m for m in self.messages if m.timestamp >= timestamp]

    def since_message(self, message_id: str) -> List[AgentMessage]:
        """Return all messages after the given message ID."""
        found = False
        result = []
        for m in self.messages:
            if m.id == message_id:
                found = True
                continue
            if found:
                result.append(m)
        return result

    def to_dicts(self) -> List[dict]:
        return [m.to_dict() for m in self.messages]
