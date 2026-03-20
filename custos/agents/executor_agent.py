"""Executor Agent — carries out actions like writing code and making API calls."""

from typing import Optional

from custos.agents.base_agent import BaseAgent
from custos.infrastructure.message_types import AgentMessage


class ExecutorAgent(BaseAgent):

    def __init__(self, llm_client, message_bus):
        super().__init__("executor", llm_client, message_bus)
        self.system_prompt = (
            "You are an execution agent. Given a plan and research findings, carry out "
            "the specified actions. This may include writing code, making API calls, "
            "generating content, or modifying files.\n\n"
            "Available tools:\n"
            "- execute_code(code): Execute a code snippet\n"
            "- call_api(endpoint, params): Call an API endpoint\n"
            "- write_file(path, content): Write content to a file\n"
            "- send_message(recipient, content): Send a message\n\n"
            "Execute the requested actions and report results clearly."
        )
        self.tools = [
            {"name": "execute_code", "description": "Execute a code snippet"},
            {"name": "call_api", "description": "Call an API endpoint"},
            {"name": "write_file", "description": "Write content to a file"},
            {"name": "send_message", "description": "Send a message to a recipient"},
        ]

    def process_message(self, message: AgentMessage) -> Optional[str]:
        """Execute actions based on the plan and research."""
        return self._invoke_llm(message.content)

    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Simulated tool execution for executor tools."""
        if tool_name == "execute_code":
            return "Code executed successfully. Output: Task completed."
        elif tool_name == "call_api":
            endpoint = kwargs.get("endpoint", "unknown")
            return f"API call to {endpoint}: 200 OK - Response received."
        elif tool_name == "write_file":
            path = kwargs.get("path", "output.txt")
            return f"File written successfully to {path}."
        elif tool_name == "send_message":
            recipient = kwargs.get("recipient", "unknown")
            return f"Message sent to {recipient}."
        return f"[Unknown tool: {tool_name}]"
