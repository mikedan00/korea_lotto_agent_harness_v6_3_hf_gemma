from __future__ import annotations

from math import comb
from typing import Iterable


def hit_count(pred: Iterable[int], actual: Iterable[int]) -> int:
    return len(set(pred) & set(actual))


def consecutive_pairs(nums: Iterable[int]) -> int:
    s = sorted(nums)
    return sum(1 for a, b in zip(s, s[1:]) if b == a + 1)


def combo_stats(nums: Iterable[int], last_numbers: Iterable[int] | None = None) -> dict:
    s = sorted(nums)
    odd = sum(n % 2 for n in s)
    high = sum(n >= 23 for n in s)
    return {
        "numbers": s,
        "sum": sum(s),
        "odd": odd,
        "even": 6 - odd,
        "high": high,
        "low": 6 - high,
        "consecutive_pairs": consecutive_pairs(s),
        "last_overlap": len(set(s) & set(last_numbers or [])),
    }


def hypergeom_match_prob(k: int, population: int = 45, success_in_population: int = 6, draws: int = 6) -> float:
    if k < 0 or k > 6:
        return 0.0
    return comb(success_in_population, k) * comb(population - success_in_population, draws - k) / comb(population, draws)


def hypergeom_tail_prob(k_min: int) -> float:
    return sum(hypergeom_match_prob(k) for k in range(k_min, 7))


def exact_probability_summary() -> dict:
    return {
        "expected_hits": 6 * 6 / 45,
        "p_0": hypergeom_match_prob(0),
        "p_1": hypergeom_match_prob(1),
        "p_2": hypergeom_match_prob(2),
        "p_3": hypergeom_match_prob(3),
        "p_4": hypergeom_match_prob(4),
        "p_5": hypergeom_match_prob(5),
        "p_6": hypergeom_match_prob(6),
        "p_ge_3": hypergeom_tail_prob(3),
        "p_ge_4": hypergeom_tail_prob(4),
    }


def brier_score_for_number_probs(probs: dict[int, float], actual: Iterable[int]) -> float:
    actual_set = set(actual)
    return sum((float(probs.get(n, 0.0)) - (1.0 if n in actual_set else 0.0)) ** 2 for n in range(1, 46)) / 45.0
