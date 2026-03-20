"""Planner Agent — decomposes tasks into subtask sequences."""

from typing import Optional

from custos.agents.base_agent import BaseAgent
from custos.infrastructure.message_types import AgentMessage


class PlannerAgent(BaseAgent):

    def __init__(self, llm_client, message_bus):
        super().__init__("planner", llm_client, message_bus)
        self.system_prompt = (
            "You are a task planning agent. Given a user request, decompose it into "
            "a sequence of subtasks. For each subtask, specify which specialist agent "
            "should handle it (Researcher, Executor, or Validator). Output a JSON "
            "plan with steps, dependencies, and assigned agents.\n\n"
            "Output format:\n"
            '{"steps": [{"id": 1, "task": "...", "agent": "researcher|executor|validator", '
            '"depends_on": []}]}'
        )
        self.tools = []  # Pure reasoning, no tools

    def process_message(self, message: AgentMessage) -> Optional[str]:
        """Decompose a task into a plan."""
        return self._invoke_llm(message.content)
