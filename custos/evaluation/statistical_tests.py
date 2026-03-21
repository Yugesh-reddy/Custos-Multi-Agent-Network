"""Statistical significance testing for experiment results."""

import numpy as np
from typing import Dict, List, Tuple


def mcnemar_test(
    results_a: List[bool],
    results_b: List[bool],
) -> Dict[str, float]:
    """McNemar's test for comparing two defenses' detection rates.

    results_a[i] = True if defense A detected attack i
    results_b[i] = True if defense B detected attack i
    """
    assert len(results_a) == len(results_b), "Results must have same length"

    # Build contingency table
    b_correct_a_wrong = 0  # B detected, A missed
    a_correct_b_wrong = 0  # A detected, B missed

    for a, b in zip(results_a, results_b):
        if b and not a:
            b_correct_a_wrong += 1
        elif a and not b:
            a_correct_b_wrong += 1

    n = b_correct_a_wrong + a_correct_b_wrong
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # McNemar statistic (with continuity correction)
    statistic = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2 / n

    # Chi-squared test with 1 df
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "b_better_count": b_correct_a_wrong,
        "a_better_count": a_correct_b_wrong,
    }


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Returns (mean, lower_bound, upper_bound).
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    means = sorted(means)
    alpha = (1 - confidence) / 2
    lower = means[int(alpha * n_bootstrap)]
    upper = means[int((1 - alpha) * n_bootstrap)]

    return (float(np.mean(arr)), float(lower), float(upper))


def wilcoxon_test(
    values_a: List[float],
    values_b: List[float],
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples.

    Compares propagation depth or other paired metrics between two defenses.
    """
    from scipy import stats

    if len(values_a) != len(values_b) or len(values_a) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    try:
        stat, p_value = stats.wilcoxon(values_a, values_b)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
