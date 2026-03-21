"""Antibody signatures for adaptive immunity layer."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AntibodySignature:
    """A learned attack signature — an 'antibody' in the immune system."""

    id: str
    name: str
    feature_extractor: str  # Name of the feature extraction method
    threshold: float        # Detection threshold
    alpha: float = 1.0      # Beta distribution param (successes + 1)
    beta_param: float = 1.0 # Beta distribution param (failures + 1)
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0.0

    @property
    def recall(self) -> float:
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def is_mature(self) -> bool:
        """True when enough data has been collected to trust this antibody."""
        return (self.alpha + self.beta_param - 2) >= 20

    @property
    def confidence_interval(self) -> tuple:
        """95% confidence interval for effectiveness from Beta distribution."""
        from scipy import stats
        low = stats.beta.ppf(0.025, self.alpha, self.beta_param)
        high = stats.beta.ppf(0.975, self.alpha, self.beta_param)
        return (low, high)

    def sample_effectiveness(self) -> float:
        """Thompson Sampling: sample from posterior Beta distribution."""
        return float(np.random.beta(self.alpha, self.beta_param))

    def update(self, was_correct_detection: bool):
        """Update antibody's belief based on outcome."""
        if was_correct_detection:
            self.alpha += 1
            self.true_positives += 1
        else:
            self.beta_param += 1
            self.false_positives += 1

    def update_negative(self, was_correct_pass: bool):
        """Update when a message was NOT flagged by this antibody."""
        if was_correct_pass:
            self.true_negatives += 1
        else:
            self.false_negatives += 1
            # Also slightly penalize the beta distribution
            self.beta_param += 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "feature_extractor": self.feature_extractor,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "beta_param": self.beta_param,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AntibodySignature":
        return cls(**data)
