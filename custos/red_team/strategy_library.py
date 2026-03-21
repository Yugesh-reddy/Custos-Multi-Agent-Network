"""Attack strategy definitions for the Red Team agent."""

from custos.attacks.direct_injection import DirectInjectionAttack
from custos.attacks.tool_poisoning import ToolPoisoningAttack
from custos.attacks.cross_infection import CrossInfectionAttack
from custos.attacks.multiturn_escalation import MultiTurnEscalationAttack
from custos.attacks.code_switching import CodeSwitchingInjectionAttack


def get_default_strategies() -> dict:
    """Return the default strategy library with Thompson Sampling parameters."""
    return {
        "direct_injection": {
            "alpha": 1.0,
            "beta": 1.0,
            "attack_class": DirectInjectionAttack,
            "description": "Direct prompt override in user input",
        },
        "tool_poisoning": {
            "alpha": 1.0,
            "beta": 1.0,
            "attack_class": ToolPoisoningAttack,
            "description": "Hidden instructions in tool descriptions/outputs",
        },
        "cross_infection": {
            "alpha": 1.0,
            "beta": 1.0,
            "attack_class": CrossInfectionAttack,
            "description": "Self-replicating payloads across agents",
        },
        "multiturn_escalation": {
            "alpha": 1.0,
            "beta": 1.0,
            "attack_class": MultiTurnEscalationAttack,
            "description": "Gradual escalation over multiple turns",
        },
        "code_switching": {
            "alpha": 1.0,
            "beta": 1.0,
            "attack_class": CodeSwitchingInjectionAttack,
            "description": "Multilingual code-switched payloads",
        },
        "hybrid_infection_codeswitching": {
            "alpha": 1.0,
            "beta": 1.0,
            "attack_class": CrossInfectionAttack,  # Uses cross-infection + code-switching
            "description": "Combine cross-infection with code-switching",
            "hybrid": True,
        },
    }
