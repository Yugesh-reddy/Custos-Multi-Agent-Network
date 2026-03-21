"""Researcher Agent — gathers information using tools and synthesizes findings."""

from typing import Optional

from custos.agents.base_agent import BaseAgent
from custos.infrastructure.message_types import AgentMessage


class ResearcherAgent(BaseAgent):

    def __init__(self, llm_client, message_bus):
        super().__init__("researcher", llm_client, message_bus)
        self.system_prompt = (
            "You are a research agent. Given a subtask, gather relevant information "
            "using your available tools. Synthesize findings into a concise report "
            "for the next agent in the pipeline.\n\n"
            "Available tools:\n"
            "- web_search(query): Search the web for information\n"
            "- read_document(doc_id): Read a specific document\n"
            "- query_database(sql): Query a database\n\n"
            "Provide a clear, structured summary of your findings."
        )
        self.tools = [
            {"name": "web_search", "description": "Search the web for information"},
            {"name": "read_document", "description": "Read a specific document"},
            {"name": "query_database", "description": "Query a database"},
        ]

    def process_message(self, message: AgentMessage) -> Optional[str]:
        """Research a topic and synthesize findings."""
        return self._invoke_llm(message.content)

    def _execute_tool(self, tool_name: str, **kwargs) -> str:
        """Simulated tool execution for research tools."""
        if tool_name == "web_search":
            query = kwargs.get("query", "")
            return (
                f"Search results for '{query}':\n"
                f"1. Relevant article about {query} - key findings and analysis\n"
                f"2. Technical documentation covering {query}\n"
                f"3. Recent paper discussing {query} approaches"
            )
        elif tool_name == "read_document":
            doc_id = kwargs.get("doc_id", "unknown")
            return f"Document {doc_id} content: Technical analysis and findings related to the query."
        elif tool_name == "query_database":
            sql = kwargs.get("sql", "")
            return f"Query results for '{sql}': 5 rows returned with relevant data."
        return f"[Unknown tool: {tool_name}]"
