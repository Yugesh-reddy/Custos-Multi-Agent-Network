"""Layer 1: Innate Immunity — fast, rule-based anomaly detection.

Runs on every inter-agent message. Must execute in <100ms. No LLM calls.
"""

import base64
import re
from typing import List, Tuple

from custos.infrastructure.message_types import AgentMessage


class InnateImmunityLayer:
    """Fast, rule-based anomaly detection layer."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?prior\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"your\s+new\s+(task|role|instruction)\s+is",
        r"you\s+are\s+now\s+a",
        r"system\s+(update|override|notice|alert)",
        r"admin\s+(override|access|command)",
        r"forward\s+(all|every|complete)\s+(message|data|context|history)",
        r"(exfiltrate|leak|extract|steal)\s+(data|credentials|keys|information)",
        r"include\s+this\s+(message|block|instruction)\s+in\s+(your|all)\s+output",
        r"<\s*IMPORTANT\s*>",
        r"\[SYSTEM\s+NOTE\]",
        r"CRITICAL\s+INSTRUCTION",
        r"MUST\s+INCLUDE\s+IN\s+ALL",
        r"do\s+not\s+(mention|tell|inform|reveal)\s+(this|the\s+user|anyone)",
    ]

    HIDDEN_MARKERS = [
        "<!--", "-->",
        "<IMPORTANT>", "</IMPORTANT>",
        "[HIDDEN]", "[/HIDDEN]",
        "```system", "```admin",
        "\x00", "\x01",
    ]

    ROLE_CHANGE_SIGNALS = [
        "you are now", "your new role", "switch to",
        "act as", "pretend to be", "from now on",
        "forget your", "abandon your",
    ]

    AUTHORITY_SIGNALS = [
        "admin", "root", "superuser", "override",
        "authorized", "elevated", "privileged",
        "security clearance", "system level",
    ]

    URGENCY_SIGNALS = [
        "immediately", "urgent", "critical",
        "time-sensitive", "do not delay",
        "failure will result", "non-compliance",
    ]

    IMPERATIVE_MARKERS = [
        "must", "should", "need to", "have to", "required to",
        "ensure", "make sure", "always", "never", "do not",
        "forward", "send", "include", "output", "execute",
        "transmit", "share", "provide", "reveal", "disclose",
    ]

    def __init__(
        self,
        block_threshold: float = 0.8,
        flag_threshold: float = 0.4,
        max_instruction_density: float = 0.3,
        max_message_length_ratio: float = 3.0,
        score_max_weight: float = 0.7,
        score_avg_weight: float = 0.3,
        base64_max_candidates: int = 50,
    ):
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold
        self.max_instruction_density = max_instruction_density
        self.max_message_length_ratio = max_message_length_ratio
        self.score_max_weight = score_max_weight
        self.score_avg_weight = score_avg_weight
        self.base64_max_candidates = base64_max_candidates

        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self.message_length_history: List[int] = []

    def analyze(self, message: AgentMessage) -> Tuple[str, float, str]:
        """Analyze a message for signs of injection/infection.

        Returns:
            verdict: "CLEAN", "FLAG", or "BLOCK"
            confidence: 0.0 to 1.0
            reason: Human-readable explanation
        """
        scores = []
        reasons = []

        # Check 1: Known injection patterns
        score, reason = self._check_patterns(message.content)
        scores.append(score)
        if reason:
            reasons.append(reason)

        # Check 2: Structural anomalies
        score, reason = self._check_structural_anomalies(message.content)
        scores.append(score)
        if reason:
            reasons.append(reason)

        # Check 3: Behavioral shift detection
        score, reason = self._check_behavioral_shift(message.content)
        scores.append(score)
        if reason:
            reasons.append(reason)

        # Check 4: Instruction density
        score, reason = self._check_instruction_density(message.content)
        scores.append(score)
        if reason:
            reasons.append(reason)

        # Aggregate
        if not scores:
            return "CLEAN", 0.0, ""

        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        final_score = self.score_max_weight * max_score + self.score_avg_weight * avg_score

        if final_score >= self.block_threshold:
            return "BLOCK", final_score, "; ".join(reasons)
        elif final_score >= self.flag_threshold:
            return "FLAG", final_score, "; ".join(reasons)
        else:
            return "CLEAN", final_score, ""

    def update_patterns(self, new_patterns: List[str]):
        """Add new detection patterns (from co-evolution learning)."""
        for pattern in new_patterns:
            self.INJECTION_PATTERNS.append(pattern)
            self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))

    def _check_patterns(self, content: str) -> Tuple[float, str]:
        matches = []
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                matches.append(pattern.pattern)

        if len(matches) >= 3:
            return 1.0, f"Multiple injection patterns detected: {len(matches)} matches"
        elif len(matches) >= 1:
            return 0.6, f"Injection pattern detected: {matches[0][:50]}"
        return 0.0, ""

    def _check_structural_anomalies(self, content: str) -> Tuple[float, str]:
        score = 0.0
        reasons = []

        # Length anomaly
        current_length = len(content)
        if self.message_length_history:
            avg_length = sum(self.message_length_history) / len(self.message_length_history)
            if avg_length > 0 and current_length / avg_length > self.max_message_length_ratio:
                score = max(score, 0.5)
                reasons.append(f"Message {current_length/avg_length:.1f}x longer than average")
        self.message_length_history.append(current_length)

        # Hidden content markers
        for marker in self.HIDDEN_MARKERS:
            if marker in content:
                score = max(score, 0.7)
                reasons.append(f"Hidden content marker: {repr(marker)}")
                break  # One marker is enough

        # Base64 detection (cap candidates for performance)
        words = content.split()
        candidates_checked = 0
        for word in words:
            if candidates_checked >= self.base64_max_candidates:
                break
            if len(word) > 20 and re.match(r"^[A-Za-z0-9+/=]+$", word):
                candidates_checked += 1
                try:
                    decoded = base64.b64decode(word).decode("utf-8", errors="ignore")
                    if any(p.search(decoded) for p in self.compiled_patterns):
                        score = max(score, 0.9)
                        reasons.append("Base64-encoded injection detected")
                        break
                except Exception:
                    pass

        return score, "; ".join(reasons)

    def _check_behavioral_shift(self, content: str) -> Tuple[float, str]:
        score = 0.0
        reasons = []
        content_lower = content.lower()

        role_hits = sum(1 for s in self.ROLE_CHANGE_SIGNALS if s in content_lower)
        if role_hits > 0:
            score = max(score, min(0.3 * role_hits, 0.9))
            reasons.append(f"Role change signals: {role_hits}")

        auth_hits = sum(1 for s in self.AUTHORITY_SIGNALS if s in content_lower)
        if auth_hits > 0:
            score = max(score, min(0.25 * auth_hits, 0.8))
            reasons.append(f"Authority claim signals: {auth_hits}")

        urgency_hits = sum(1 for s in self.URGENCY_SIGNALS if s in content_lower)
        if urgency_hits > 0:
            score = max(score, min(0.15 * urgency_hits, 0.5))
            reasons.append(f"Urgency pressure signals: {urgency_hits}")

        return score, "; ".join(reasons)

    def _check_instruction_density(self, content: str) -> Tuple[float, str]:
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return 0.0, ""

        imperative_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(marker in sentence_lower for marker in self.IMPERATIVE_MARKERS):
                imperative_count += 1

        density = imperative_count / len(sentences)
        if density > self.max_instruction_density:
            return min(density * 1.5, 1.0), f"High instruction density: {density:.2f}"
        return 0.0, ""
