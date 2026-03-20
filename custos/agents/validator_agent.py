"""Validator Agent — reviews outputs for correctness and safety."""

from typing import Optional

from custos.agents.base_agent import BaseAgent
from custos.infrastructure.message_types import AgentMessage


class ValidatorAgent(BaseAgent):

    def __init__(self, llm_client, message_bus):
        super().__init__("validator", llm_client, message_bus)
        self.system_prompt = (
            "You are a validation agent. Review the outputs from the Executor agent. "
            "Check for correctness, safety, and adherence to the original request. "
            "Flag any issues and provide a pass/fail verdict with explanation.\n\n"
            "Available tools:\n"
            "- verify_output(output, criteria): Verify output against criteria\n\n"
            "Provide a clear verdict: PASS or FAIL, with a brief explanation."
        )
        self.tools = [
            {"name": "verify_output", "description": "Verify output against criteria"},
        ]

    def process_message(self, message: AgentMessage) -> Optional[str]:
        """Validate the executor's output."""
        return self._invoke_llm(message.content)

    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Simulated tool execution for validator tools."""
        if tool_name == "verify_output":
            return "Verification complete: Output meets specified criteria. PASS."
        return f"[Unknown tool: {tool_name}]"
