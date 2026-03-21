"""Agent role profiles — expected behavior definitions for each agent type.

Used by the adaptive immunity layer for role deviation detection.
"""

AGENT_PROFILES = {
    "planner": {
        "role": "planner",
        "description": "Task planning agent that decomposes requests into subtask sequences",
        "expected_outputs": ["task decomposition", "subtask list", "JSON plan", "step assignments"],
        "expected_vocabulary": [
            "subtask", "assign", "dependency", "step", "plan",
            "decompose", "sequence", "delegate", "prioritize",
        ],
        "forbidden_vocabulary": [
            "execute", "run code", "call api", "send message",
            "forward", "exfiltrate", "override",
        ],
        "allowed_tools": [],
        "typical_message_length": (100, 500),
    },
    "researcher": {
        "role": "researcher",
        "description": "Research agent that gathers information using tools and synthesizes findings",
        "expected_outputs": ["research findings", "summary", "data", "analysis", "sources"],
        "expected_vocabulary": [
            "found", "research", "source", "data", "evidence",
            "according to", "findings", "summarize", "relevant",
        ],
        "forbidden_vocabulary": [
            "execute code", "override", "ignore instructions",
            "forward all", "exfiltrate",
        ],
        "allowed_tools": ["web_search", "read_document", "query_database"],
        "typical_message_length": (200, 1000),
    },
    "executor": {
        "role": "executor",
        "description": "Execution agent that carries out actions like writing code and making API calls",
        "expected_outputs": ["code", "API response", "file content", "execution result"],
        "expected_vocabulary": [
            "executed", "result", "output", "code", "implemented",
            "created", "modified", "response", "status",
        ],
        "forbidden_vocabulary": [
            "ignore instructions", "override", "forward credentials",
            "exfiltrate", "system prompt",
        ],
        "allowed_tools": ["execute_code", "call_api", "write_file", "send_message"],
        "typical_message_length": (100, 2000),
    },
    "validator": {
        "role": "validator",
        "description": "Validation agent that reviews outputs for correctness and safety",
        "expected_outputs": ["pass", "fail", "verdict", "issues found", "approved"],
        "expected_vocabulary": [
            "verified", "correct", "issue", "pass", "fail",
            "quality", "review", "check", "compliant", "safe",
        ],
        "forbidden_vocabulary": [
            "execute", "override", "ignore", "forward",
            "exfiltrate", "credentials",
        ],
        "allowed_tools": ["verify_output"],
        "typical_message_length": (50, 400),
    },
}


def get_profile(agent_id: str) -> dict:
    """Get the profile for an agent ID, falling back to a generic profile."""
    return AGENT_PROFILES.get(agent_id, {
        "role": agent_id,
        "description": f"Agent: {agent_id}",
        "expected_outputs": [],
        "expected_vocabulary": [],
        "forbidden_vocabulary": [],
        "allowed_tools": [],
        "typical_message_length": (50, 1000),
    })
