"""Feature extraction for adaptive immunity layer.

Extracts numerical features from messages for antibody matching.
"""

import re
import unicodedata
from typing import Dict

from custos.infrastructure.message_types import AgentMessage


class FeatureExtractor:
    """Extracts features from inter-agent messages for anomaly detection."""

    def __init__(self, use_embeddings: bool = False):
        self._encoder = None
        self._role_embeddings: Dict[str, object] = {}
        self._use_embeddings = use_embeddings
        if use_embeddings:
            self._load_encoder()

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self._use_embeddings = False

    def extract_all(
        self,
        message: AgentMessage,
        agent_profiles: Dict,
    ) -> Dict[str, float]:
        """Extract all features from a message. Returns feature name -> score [0.0, 1.0]."""
        content = message.content
        profile = agent_profiles.get(message.sender, {})

        return {
            "semantic_coherence": self.semantic_coherence(content, profile.get("role", "")),
            "instruction_ratio": self.instruction_ratio(content),
            "context_leakage_score": self.context_leakage_score(content),
            "replication_score": self.replication_score(content),
            "role_deviation_score": self.role_deviation_score(content, profile),
            "multilingual_score": self.multilingual_score(content),
            "exfiltration_score": self.exfiltration_score(content),
            "authority_score": self.authority_score(content),
        }

    def semantic_coherence(self, content: str, role: str) -> float:
        """Measure semantic coherence between content and sender's role.

        High score = low coherence = anomalous.
        Uses embeddings if available, otherwise keyword overlap.
        """
        if not role:
            return 0.0

        if self._use_embeddings and self._encoder is not None:
            return self._semantic_coherence_embeddings(content, role)

        return self._semantic_coherence_keywords(content, role)

    def _semantic_coherence_keywords(self, content: str, role: str) -> float:
        """Keyword-based semantic coherence (lightweight fallback)."""
        role_keywords = {
            "planner": {"plan", "step", "subtask", "assign", "decompose", "sequence", "dependency"},
            "researcher": {"found", "research", "source", "data", "evidence", "findings", "summary"},
            "executor": {"executed", "result", "output", "code", "implemented", "created", "response"},
            "validator": {"verified", "correct", "issue", "pass", "fail", "quality", "review"},
        }

        expected = role_keywords.get(role, set())
        if not expected:
            return 0.0

        content_words = set(content.lower().split())
        overlap = len(content_words & expected)
        coherence = overlap / max(len(expected), 1)
        # Invert: low coherence = high anomaly score
        return max(0.0, min(1.0 - coherence, 1.0))

    def _semantic_coherence_embeddings(self, content: str, role: str) -> float:
        """Embedding-based semantic coherence."""
        import numpy as np

        if role not in self._role_embeddings:
            role_text = f"As a {role} agent, I would typically say:"
            self._role_embeddings[role] = self._encoder.encode(role_text)

        role_emb = self._role_embeddings[role]
        content_emb = self._encoder.encode(content[:512])

        # Cosine similarity
        sim = np.dot(role_emb, content_emb) / (
            np.linalg.norm(role_emb) * np.linalg.norm(content_emb) + 1e-8
        )
        return max(0.0, min(1.0 - float(sim), 1.0))

    def instruction_ratio(self, content: str) -> float:
        """Ratio of imperative/instruction sentences to total."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
        if not sentences:
            return 0.0

        imperative_words = {
            "must", "should", "need", "ensure", "forward",
            "send", "include", "execute", "output", "always",
            "never", "required", "mandatory", "transmit",
        }
        imperative_count = sum(
            1 for s in sentences
            if any(w in s.lower().split() for w in imperative_words)
        )
        return min(imperative_count / len(sentences), 1.0)

    def context_leakage_score(self, content: str) -> float:
        """Detect if message contains leaked system prompts or credentials."""
        leakage_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",
            r"ANTHROPIC_API_KEY",
            r"OPENAI_API_KEY",
            r"system\s*prompt\s*[:=]",
            r"you\s+are\s+a\s+\w+\s+agent",
            r"(password|secret|token)\s*[:=]\s*\S+",
            r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",
            r"ssh-rsa\s+",
        ]
        hits = sum(1 for p in leakage_patterns if re.search(p, content, re.IGNORECASE))
        return min(hits * 0.3, 1.0)

    def replication_score(self, content: str) -> float:
        """Detect self-replication instructions."""
        replication_phrases = [
            "include this", "copy this", "forward this",
            "pass this along", "repeat this", "append this",
            "in your output", "in all outputs", "to downstream",
            "include verbatim", "must be included",
            "include the full text", "in their next outgoing",
        ]
        hits = sum(1 for p in replication_phrases if p.lower() in content.lower())
        return min(hits * 0.25, 1.0)

    def role_deviation_score(self, content: str, profile: Dict) -> float:
        """Detect if sender is acting outside their defined role."""
        forbidden = profile.get("forbidden_vocabulary", [])
        if not forbidden:
            return 0.0

        content_lower = content.lower()
        hits = sum(1 for word in forbidden if word.lower() in content_lower)
        return min(hits * 0.3, 1.0)

    def multilingual_score(self, content: str) -> float:
        """Detect unexpected non-English tokens in an English-dominant system."""
        total_chars = len(content)
        if total_chars == 0:
            return 0.0

        non_latin_chars = sum(
            1 for c in content
            if unicodedata.category(c).startswith("L") and not c.isascii()
        )
        ratio = non_latin_chars / total_chars
        if ratio > 0.05:
            return min(ratio * 5, 1.0)
        return 0.0

    def exfiltration_score(self, content: str) -> float:
        """Detect data exfiltration intent."""
        exfil_patterns = [
            r"https?://\S+",
            r"\S+@\S+\.\S+",
            r"send_message\(",
            r"call_api\(",
            r"forward.*to.*external",
            r"upload.*to",
            r"post.*to.*endpoint",
        ]
        hits = sum(1 for p in exfil_patterns if re.search(p, content, re.IGNORECASE))
        return min(hits * 0.3, 1.0)

    def authority_score(self, content: str) -> float:
        """Detect authority escalation claims."""
        authority_patterns = [
            "admin override", "elevated privileges", "security clearance",
            "authorized to", "root access", "system level", "superuser",
            "compliance requirement", "mandatory audit", "by order of",
        ]
        hits = sum(1 for p in authority_patterns if p.lower() in content.lower())
        return min(hits * 0.35, 1.0)
