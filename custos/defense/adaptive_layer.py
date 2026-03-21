"""Layer 2: Adaptive Immunity — Thompson Sampling over antibody signatures.

Core novel contribution: treats detection as a multi-armed bandit problem.
Each antibody is an 'arm'. Thompson Sampling balances exploration vs exploitation.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

from custos.defense.antibody_library import AntibodySignature
from custos.defense.feature_extractors import FeatureExtractor
from custos.infrastructure.message_types import AgentMessage


class AdaptiveImmunityLayer:
    """Thompson Sampling-based adaptive detection layer."""

    def __init__(
        self,
        top_k: int = 4,
        use_embeddings: bool = False,
    ):
        self.top_k = top_k
        self.feature_extractor = FeatureExtractor(use_embeddings=use_embeddings)
        self.antibody_library: List[AntibodySignature] = []
        self._initialize_antibodies()

    def _initialize_antibodies(self):
        """Initialize the starting antibody library."""
        self.antibody_library = [
            AntibodySignature(
                id="ab_semantic_shift",
                name="Semantic Coherence Shift",
                feature_extractor="semantic_coherence",
                threshold=0.3,
            ),
            AntibodySignature(
                id="ab_instruction_ratio",
                name="Instruction-to-Content Ratio",
                feature_extractor="instruction_ratio",
                threshold=0.4,
            ),
            AntibodySignature(
                id="ab_context_leakage",
                name="Context Leakage Detector",
                feature_extractor="context_leakage_score",
                threshold=0.5,
            ),
            AntibodySignature(
                id="ab_replication_marker",
                name="Self-Replication Pattern",
                feature_extractor="replication_score",
                threshold=0.3,
            ),
            AntibodySignature(
                id="ab_role_deviation",
                name="Role Deviation Detector",
                feature_extractor="role_deviation_score",
                threshold=0.4,
            ),
            AntibodySignature(
                id="ab_multilingual_anomaly",
                name="Multilingual Anomaly Detector",
                feature_extractor="multilingual_score",
                threshold=0.3,
            ),
            AntibodySignature(
                id="ab_exfiltration_intent",
                name="Data Exfiltration Intent",
                feature_extractor="exfiltration_score",
                threshold=0.5,
            ),
            AntibodySignature(
                id="ab_authority_escalation",
                name="Authority Escalation Detector",
                feature_extractor="authority_score",
                threshold=0.4,
            ),
        ]

    def analyze(
        self,
        message: AgentMessage,
        agent_profiles: Dict,
    ) -> Tuple[str, float, str, List[str]]:
        """Run adaptive analysis using Thompson Sampling.

        Returns:
            verdict: "CLEAN", "FLAG", or "BLOCK"
            confidence: 0.0 to 1.0
            reason: Human-readable explanation
            active_antibody_ids: IDs of antibodies that were evaluated
        """
        features = self.feature_extractor.extract_all(message, agent_profiles)

        # Thompson Sampling: select top-K antibodies
        K = min(self.top_k, len(self.antibody_library))
        sampled_values = [
            (ab, ab.sample_effectiveness())
            for ab in self.antibody_library
        ]
        sampled_values.sort(key=lambda x: x[1], reverse=True)
        selected = [ab for ab, _ in sampled_values[:K]]
        active_ids = [ab.id for ab in selected]

        # Run selected antibodies
        detections = []
        for antibody in selected:
            feature_value = features.get(antibody.feature_extractor, 0.0)
            if feature_value > antibody.threshold:
                detections.append((antibody, feature_value))

        # Aggregate results
        if len(detections) >= 2:
            confidence = max(fv for _, fv in detections)
            reasons = [f"{ab.name}: {fv:.2f}" for ab, fv in detections]
            return "BLOCK", confidence, "; ".join(reasons), active_ids
        elif len(detections) == 1:
            ab, fv = detections[0]
            return "FLAG", fv, f"{ab.name}: {fv:.2f}", active_ids
        else:
            return "CLEAN", 0.0, "", active_ids

    def provide_feedback(self, antibody_id: str, was_correct: bool):
        """Update antibody belief after ground truth is known."""
        for ab in self.antibody_library:
            if ab.id == antibody_id:
                ab.update(was_correct)
                break

    def evolve_antibodies(self, attack_examples: List[Dict]):
        """Create new antibodies from observed attack patterns.

        Called after a novel attack evades existing antibodies.
        """
        for example in attack_examples:
            new_antibody = AntibodySignature(
                id=f"ab_evolved_{len(self.antibody_library)}",
                name=f"Evolved: {example.get('attack_type', 'unknown')}",
                feature_extractor=example.get("feature_key", "instruction_ratio"),
                threshold=example.get("suggested_threshold", 0.3),
                alpha=2.0,  # Slight positive prior (we know it targets a real attack)
                beta_param=1.0,
            )
            self.antibody_library.append(new_antibody)

    def select_antibodies_ucb1(self, K: int = 4) -> List[AntibodySignature]:
        """UCB1 fallback if Thompson Sampling doesn't converge."""
        total_trials = sum(
            ab.alpha + ab.beta_param - 2 for ab in self.antibody_library
        )
        total_trials = max(total_trials, 1)

        scores = []
        for ab in self.antibody_library:
            n_i = ab.alpha + ab.beta_param - 2
            if n_i == 0:
                scores.append((ab, float("inf")))
            else:
                mean = ab.alpha / (ab.alpha + ab.beta_param)
                ucb = mean + math.sqrt(2 * math.log(total_trials) / n_i)
                scores.append((ab, ucb))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [ab for ab, _ in scores[:K]]

    def get_library_stats(self) -> Dict:
        """Get statistics about the antibody library."""
        return {
            "total_antibodies": len(self.antibody_library),
            "mature_antibodies": sum(1 for ab in self.antibody_library if ab.is_mature),
            "antibodies": [
                {
                    "id": ab.id,
                    "name": ab.name,
                    "precision": ab.precision,
                    "recall": ab.recall,
                    "alpha": ab.alpha,
                    "beta": ab.beta_param,
                }
                for ab in self.antibody_library
            ],
        }
