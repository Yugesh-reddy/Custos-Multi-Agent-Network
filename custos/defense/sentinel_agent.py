"""Sentinel Agent — the main defense orchestrator.

Processing pipeline:
1. Innate Layer (fast, <100ms) → catches obvious attacks
2. If innate says FLAG → Adaptive Layer (slower, uses ML features)
3. If either says BLOCK → Quarantine Controller isolates the source
4. Feedback loop: ground truth updates adaptive layer's Thompson Sampling
"""

import logging
from typing import Dict, List, Optional

from custos.defense.innate_layer import InnateImmunityLayer
from custos.defense.adaptive_layer import AdaptiveImmunityLayer
from custos.defense.quarantine_controller import QuarantineController
from custos.infrastructure.message_bus import MessageBus
from custos.infrastructure.message_types import AgentMessage, ThreatLevel

logger = logging.getLogger(__name__)


class SentinelAgent:
    """The main defense orchestrator — hooks into the MessageBus."""

    def __init__(
        self,
        message_bus: MessageBus,
        agent_profiles: Dict,
        use_embeddings: bool = False,
    ):
        self.bus = message_bus
        self.agent_profiles = agent_profiles

        # Initialize immune layers
        self.innate = InnateImmunityLayer()
        self.adaptive = AdaptiveImmunityLayer(use_embeddings=use_embeddings)
        self.quarantine = QuarantineController(message_bus)

        # Track which antibodies were active per message (for feedback)
        self._pending_verdicts: Dict[str, List[str]] = {}  # message_id -> [antibody_ids]

        # Statistics
        self.stats = {
            "messages_inspected": 0,
            "innate_blocks": 0,
            "innate_flags": 0,
            "adaptive_blocks": 0,
            "adaptive_flags": 0,
            "quarantines_executed": 0,
            "true_positives": 0,
            "false_positives": 0,
        }

        # Register with message bus
        self.bus.register_interceptor(self.inspect_message)

    def inspect_message(self, message: AgentMessage) -> str:
        """Main inspection entry point. Called by MessageBus for every message.

        Returns: "CLEAN", "FLAG", or "BLOCK"
        """
        self.stats["messages_inspected"] += 1

        # Check if sender is under enhanced monitoring
        is_monitored = self.quarantine.is_enhanced_monitoring(message.sender)
        if is_monitored:
            self.quarantine.decrement_monitoring(message.sender)

        # Layer 1: Innate immunity (fast)
        innate_verdict, innate_confidence, innate_reason = self.innate.analyze(message)

        if innate_verdict == "BLOCK":
            self.stats["innate_blocks"] += 1
            message.threat_assessment = ThreatLevel.INFECTED
            message.sentinel_notes = f"Innate BLOCK: {innate_reason}"

            # Execute quarantine
            self.quarantine.execute_quarantine(
                agent_id=message.sender,
                threat_level="BLOCK",
                reason=innate_reason,
                message_id=message.id,
            )
            self.stats["quarantines_executed"] += 1

            logger.info(
                f"SENTINEL BLOCK (innate): {message.sender}→{message.receiver} | {innate_reason}"
            )
            return "BLOCK"

        # Layer 2: Adaptive immunity (for FLAG or monitored agents)
        if innate_verdict == "FLAG" or is_monitored:
            if innate_verdict == "FLAG":
                self.stats["innate_flags"] += 1

            adaptive_verdict, adaptive_confidence, adaptive_reason, active_ids = (
                self.adaptive.analyze(message, self.agent_profiles)
            )

            # Track active antibodies for feedback
            self._pending_verdicts[message.id] = active_ids

            if adaptive_verdict == "BLOCK":
                self.stats["adaptive_blocks"] += 1
                message.threat_assessment = ThreatLevel.INFECTED
                combined_reason = f"Innate: {innate_reason}; Adaptive: {adaptive_reason}"
                message.sentinel_notes = f"Adaptive BLOCK: {combined_reason}"

                self.quarantine.execute_quarantine(
                    agent_id=message.sender,
                    threat_level="BLOCK",
                    reason=combined_reason,
                    message_id=message.id,
                )
                self.stats["quarantines_executed"] += 1

                logger.info(
                    f"SENTINEL BLOCK (adaptive): {message.sender}→{message.receiver} | {adaptive_reason}"
                )
                return "BLOCK"

            elif adaptive_verdict == "FLAG":
                self.stats["adaptive_flags"] += 1
                message.threat_assessment = ThreatLevel.SUSPICIOUS
                message.sentinel_notes = f"Adaptive FLAG: {adaptive_reason}"

                # Activate enhanced monitoring if not already
                if not is_monitored:
                    self.quarantine.activate_enhanced_monitoring(message.sender)

                logger.info(
                    f"SENTINEL FLAG: {message.sender}→{message.receiver} | {adaptive_reason}"
                )
                return "FLAG"

        # Clean
        return "CLEAN"

    def receive_ground_truth(self, message_id: str, was_attack: bool):
        """After ground truth is known, update adaptive layer's Thompson Sampling."""
        if was_attack:
            self.stats["true_positives"] += 1
        else:
            self.stats["false_positives"] += 1

        # Update antibodies that were active for this message
        active_ids = self._pending_verdicts.pop(message_id, [])
        for ab_id in active_ids:
            self.adaptive.provide_feedback(ab_id, was_attack)

    def get_performance_report(self) -> Dict:
        """Get performance statistics."""
        total_detections = self.stats["innate_blocks"] + self.stats["adaptive_blocks"]
        return {
            **self.stats,
            "total_detections": total_detections,
            "detection_rate": total_detections / max(self.stats["messages_inspected"], 1),
            "antibody_library_size": len(self.adaptive.antibody_library),
            "quarantined_agents": sorted(self.bus.quarantined_agents),
            "enhanced_monitoring": dict(self.quarantine.enhanced_monitoring),
        }
