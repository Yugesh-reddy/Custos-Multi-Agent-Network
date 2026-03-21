"""Attack success metrics."""

from dataclasses import dataclass


@dataclass
class AttackMetrics:
    # Primary metrics
    attack_success_rate: float = 0.0       # % of attacks that achieved their goal
    propagation_depth: float = 0.0         # Average number of agents infected per attack
    propagation_speed: float = 0.0         # Messages until first propagation
    data_exfiltrated: bool = False         # Whether sensitive data was leaked
    exfiltration_volume: int = 0           # Bytes of data leaked

    # Secondary metrics
    detection_evasion_rate: float = 0.0    # % of attacks that evaded defenses
    behavioral_deviation: float = 0.0      # How much agent behavior changed
    task_disruption_rate: float = 0.0      # % of legitimate tasks that failed
    stealth_score: float = 0.0             # How detectable the attack was (lower = stealthier)

    def to_dict(self) -> dict:
        return {
            "attack_success_rate": self.attack_success_rate,
            "propagation_depth": self.propagation_depth,
            "propagation_speed": self.propagation_speed,
            "data_exfiltrated": self.data_exfiltrated,
            "exfiltration_volume": self.exfiltration_volume,
            "detection_evasion_rate": self.detection_evasion_rate,
            "behavioral_deviation": self.behavioral_deviation,
            "task_disruption_rate": self.task_disruption_rate,
            "stealth_score": self.stealth_score,
        }
