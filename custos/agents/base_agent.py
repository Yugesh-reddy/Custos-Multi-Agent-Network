"""Abstract base class for all agents in the multi-agent network."""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from custos.infrastructure.message_bus import MessageBus
from custos.infrastructure.message_types import AgentMessage, MessageType
from custos.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all worker agents."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        message_bus: MessageBus,
    ):
        self.agent_id = agent_id
        self.llm = llm_client
        self.bus = message_bus
        self.system_prompt: str = ""
        self.tools: List[Dict] = []
        self.memory: List[Dict] = []  # conversation history
        self.max_memory: int = 20  # keep last N exchanges
        self.bus.register_agent(self.agent_id, self)

    @abstractmethod
    def process_message(self, message: AgentMessage) -> Optional[str]:
        """Process an incoming message and return a response string.

        Subclasses implement their specific logic here.
        Returns the response content string, or None if no response.
        """

    def receive_and_respond(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive a message, process it, and send the response.

        This is the main entry point called by the TaskRunner.
        Returns the response AgentMessage if sent successfully, None otherwise.
        """
        # Snapshot state before processing (for rollback)
        self.bus.snapshot_agent_state(self.agent_id, self.snapshot_state())

        # Process the message
        response_content = self.process_message(message)
        if response_content is None:
            return None

        # Add to memory
        self.memory.append({"role": "user", "content": message.content})
        self.memory.append({"role": "assistant", "content": response_content})

        # Trim memory if too long
        if len(self.memory) > self.max_memory * 2:
            self.memory = self.memory[-(self.max_memory * 2):]

        return response_content

    def _build_llm_messages(self, incoming_content: str) -> List[Dict]:
        """Build the messages list for an LLM call."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation memory
        for mem in self.memory:
            messages.append(mem)

        # Add the incoming message
        messages.append({"role": "user", "content": incoming_content})
        return messages

    def _invoke_llm(self, incoming_content: str) -> str:
        """Build messages and call the LLM."""
        messages = self._build_llm_messages(incoming_content)
        return self.llm.invoke(messages)

    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a simulated tool. Override in subclasses for specific tools."""
        return f"[Tool {tool_name} executed with args: {kwargs}]"

    def snapshot_state(self) -> dict:
        """Return serializable state for rollback."""
        return {
            "agent_id": self.agent_id,
            "memory": copy.deepcopy(self.memory),
        }

    def restore_state(self, state: dict):
        """Restore from a snapshot."""
        self.memory = copy.deepcopy(state.get("memory", []))

    def reset(self):
        """Reset agent state for a new task."""
        self.memory = []
