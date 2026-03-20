"""Unified LLM client — 4 providers, 6+ models, one interface."""

import os
import time
import logging
from typing import List, Optional

from custos.configs.config import load_model_registry

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for Ollama, Azure OpenAI, and AWS Bedrock."""

    def __init__(
        self,
        provider_name: str,
        dry_run: bool = False,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        registry = load_model_registry()
        if provider_name not in registry:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {list(registry.keys())}"
            )

        cfg = registry[provider_name]
        self.provider_name = provider_name
        self.type = cfg["type"]
        self.model = cfg["model"]
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.cost_per_1k_input = cfg.get("cost_per_1k_input", 0.0)
        self.cost_per_1k_output = cfg.get("cost_per_1k_output", 0.0)
        self.total_cost = 0.0
        self.total_calls = 0

        self._client = None
        self._bedrock = None

        if not dry_run:
            self._init_client(cfg)

    def _init_client(self, cfg: dict):
        if self.type == "ollama":
            from openai import OpenAI

            self._client = OpenAI(
                base_url=cfg.get("base_url", "http://localhost:11434/v1"),
                api_key="ollama",
            )
        elif self.type == "azure":
            from openai import OpenAI

            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            key = os.environ.get("AZURE_OPENAI_KEY", "")
            if not endpoint or not key:
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set"
                )
            self._client = OpenAI(
                base_url=f"{endpoint}/openai/deployments/{cfg['deployment']}",
                api_key=key,
                default_headers={"api-version": "2024-10-21"},
            )
        elif self.type == "bedrock":
            import boto3

            region = cfg.get("region", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            self._bedrock = boto3.client("bedrock-runtime", region_name=region)

    def invoke(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Send messages to the LLM and return the response text."""
        if self.dry_run:
            self.total_calls += 1
            return f"[DRY_RUN: {self.provider_name}/{self.model}] Processed {len(messages)} messages"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = self._call(messages, max_tokens, temperature)
                self.total_calls += 1
                self._track_cost(messages, result)
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: {last_error}"
        )

    def _call(self, messages: List[dict], max_tokens: int, temperature: float) -> str:
        if self.type in ("ollama", "azure"):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        elif self.type == "bedrock":
            bedrock_messages = [
                {
                    "role": m["role"],
                    "content": [{"text": m["content"]}],
                }
                for m in messages
                if m["role"] != "system"
            ]
            system_msgs = [m for m in messages if m["role"] == "system"]
            kwargs = {
                "modelId": self.model,
                "messages": bedrock_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            }
            if system_msgs:
                kwargs["system"] = [{"text": system_msgs[0]["content"]}]
            resp = self._bedrock.converse(**kwargs)
            return resp["output"]["message"]["content"][0]["text"]
        else:
            raise ValueError(f"Unknown provider type: {self.type}")

    def _track_cost(self, messages: List[dict], response: str):
        """Estimate and accumulate cost based on character count approximation."""
        input_chars = sum(len(m.get("content", "")) for m in messages)
        output_chars = len(response)
        # Rough token estimate: ~4 chars per token
        input_tokens = input_chars / 4
        output_tokens = output_chars / 4
        cost = (
            (input_tokens / 1000) * self.cost_per_1k_input
            + (output_tokens / 1000) * self.cost_per_1k_output
        )
        self.total_cost += cost

    def get_cost_report(self) -> dict:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost, 4),
        }
