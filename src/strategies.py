from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import numpy as np
import pandas as pd

from .data_loader import Draw


def _z(scores: dict[int, float]) -> dict[int, float]:
    vals = np.array([scores.get(n, 0.0) for n in range(1, 46)], dtype=float)
    mu = float(vals.mean())
    sd = float(vals.std()) or 1.0
    return {n: (scores.get(n, 0.0) - mu) / sd for n in range(1, 46)}


def _minmax_positive(scores: dict[int, float]) -> dict[int, float]:
    vals = np.array([scores.get(n, 0.0) for n in range(1, 46)], dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {n: 1.0 for n in range(1, 46)}
    return {n: 0.05 + 0.95 * (scores.get(n, 0.0) - lo) / (hi - lo) for n in range(1, 46)}


@dataclass
class StrategyOutput:
    name: str
    scores: dict[int, float]
    explanation: str


class BaseStrategy:
    name = "base"
    explanation = "base strategy"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        raise NotImplementedError


class LongFrequencyStrategy(BaseStrategy):
    name = "long_frequency"
    explanation = "전체 이력의 장기 출현 빈도 z-score"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        c = Counter(n for d in draws for n in d.numbers)
        return StrategyOutput(self.name, _z({n: c[n] for n in range(1, 46)}), self.explanation)


class RecentDecayStrategy(BaseStrategy):
    name = "recent_decay"
    explanation = "최근 회차에 더 큰 가중치를 주는 지수 감쇠 빈도"
    def __init__(self, half_life: float = 40.0):
        self.half_life = half_life
    def score(self, draws: list[Draw]) -> StrategyOutput:
        latest = draws[-1].round
        scores = {n: 0.0 for n in range(1, 46)}
        for d in draws:
            age = max(0, latest - d.round)
            w = 0.5 ** (age / self.half_life)
            for n in d.numbers:
                scores[n] += w
        return StrategyOutput(self.name, _z(scores), self.explanation)


class GapStrategy(BaseStrategy):
    name = "gap"
    explanation = "현재 미출현 간격 / 평균 출현 간격 기반 overdue 점수"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        latest = draws[-1].round
        by_num = {n: [] for n in range(1, 46)}
        for d in draws:
            for n in d.numbers:
                by_num[n].append(d.round)
        scores = {}
        for n, rounds in by_num.items():
            if len(rounds) <= 1:
                scores[n] = 0.0
                continue
            gaps = [b - a for a, b in zip(rounds, rounds[1:])]
            avg = sum(gaps) / max(1, len(gaps))
            cur = latest - rounds[-1]
            scores[n] = cur / (avg + 1e-9)
        return StrategyOutput(self.name, _z(scores), self.explanation)


class HotColdStrategy(BaseStrategy):
    name = "hot_cold"
    explanation = "최근 30회 hot 신호와 장기 과소/과다 출현 균형"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        recent = draws[-30:] if len(draws) >= 30 else draws
        c_recent = Counter(n for d in recent for n in d.numbers)
        c_all = Counter(n for d in draws for n in d.numbers)
        zr = _z({n: c_recent[n] for n in range(1, 46)})
        zl = _z({n: c_all[n] for n in range(1, 46)})
        return StrategyOutput(self.name, _z({n: 0.75 * zr[n] + 0.25 * zl[n] for n in range(1, 46)}), self.explanation)


class MomentumStrategy(BaseStrategy):
    name = "momentum"
    explanation = "최근 30회 출현률 - 전체 출현률"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        recent = draws[-30:] if len(draws) >= 30 else draws
        c_recent = Counter(n for d in recent for n in d.numbers)
        c_all = Counter(n for d in draws for n in d.numbers)
        scores = {}
        for n in range(1, 46):
            scores[n] = c_recent[n] / max(1, len(recent)) - c_all[n] / max(1, len(draws))
        return StrategyOutput(self.name, _z(scores), self.explanation)


class BayesStrategy(BaseStrategy):
    name = "bayes"
    explanation = "Beta-Binomial 사후평균, 약한 균등 prior 포함"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        # Expected base rate per number per draw = 6/45. Prior equivalent 30 draws.
        alpha0 = 30 * 6 / 45
        beta0 = 30 * (1 - 6 / 45)
        c = Counter(n for d in draws for n in d.numbers)
        scores = {n: (alpha0 + c[n]) / (alpha0 + beta0 + len(draws)) for n in range(1, 46)}
        return StrategyOutput(self.name, _z(scores), self.explanation)


class PairAffinityStrategy(BaseStrategy):
    name = "pair_affinity"
    explanation = "번호별 평균 pair co-occurrence affinity"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        mat = pair_matrix(draws)
        scores = {n: float(sum(mat[n].values()) / 44.0) for n in range(1, 46)}
        return StrategyOutput(self.name, _z(scores), self.explanation)


class AntiCrowdStrategy(BaseStrategy):
    name = "anti_crowd"
    explanation = "직전 회차 반복과 생일수 편향을 완화하는 contrarian 보정"
    def score(self, draws: list[Draw]) -> StrategyOutput:
        last = set(draws[-1].numbers)
        scores = {}
        for n in range(1, 46):
            s = 0.0
            if n in last:
                s -= 0.8
            if n > 31:
                s += 0.15
            if n in [1, 7, 11, 13, 17, 23, 29, 31, 37, 41, 43]:
                s += 0.05
            scores[n] = s
        return StrategyOutput(self.name, _z(scores), self.explanation)



# -----------------------------------------------------------------------------
# Prime factorization based research strategy
# -----------------------------------------------------------------------------
# 목적:
#   회차별 당첨번호 6개를 소인수분해하여, 숫자 자체의 빈도뿐 아니라
#   "소인수 성분의 출현/미출현/최근성/조합 구조"를 별도 신호로 본다.
# 주의:
#   이 전략은 연구용 feature engineering이다. 공정 추첨의 물리적 확률을
#   바꾸지는 않으며, rolling-forward 백테스트에서만 신뢰도를 판단한다.

PRIME_BASIS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
PRIME_SET = set(PRIME_BASIS)


def prime_factor_exponents(n: int) -> dict[int, int]:
    """1~45 정수를 PRIME_BASIS 기준의 소인수 지수 벡터로 변환한다."""
    if n < 1 or n > 45:
        raise ValueError(f"로또 번호 범위가 아닙니다: {n}")
    x = n
    out = {p: 0 for p in PRIME_BASIS}
    for p in PRIME_BASIS:
        while x > 1 and x % p == 0:
            out[p] += 1
            x //= p
    return out


FACTOR_TABLE = {n: prime_factor_exponents(n) for n in range(1, 46)}
OMEGA_TABLE = {n: sum(FACTOR_TABLE[n].values()) for n in range(1, 46)}
DISTINCT_FACTOR_TABLE = {n: sum(1 for v in FACTOR_TABLE[n].values() if v > 0) for n in range(1, 46)}
SQUAREFREE_TABLE = {n: int(all(v <= 1 for v in FACTOR_TABLE[n].values())) for n in range(1, 46)}


def draw_prime_profile(numbers: tuple[int, ...] | list[int]) -> dict[str, int | float]:
    """당첨번호 6개 조합을 소수/합성수/소인수 분포 관점으로 요약한다."""
    nums = list(numbers)
    prime_count = sum(1 for n in nums if n in PRIME_SET)
    one_count = sum(1 for n in nums if n == 1)
    composite_count = 6 - prime_count - one_count
    omega_sum = sum(OMEGA_TABLE[n] for n in nums)
    distinct_factor_sum = sum(DISTINCT_FACTOR_TABLE[n] for n in nums)
    squarefree_count = sum(SQUAREFREE_TABLE[n] for n in nums)
    divisible_2 = sum(1 for n in nums if n % 2 == 0)
    divisible_3 = sum(1 for n in nums if n % 3 == 0)
    divisible_5 = sum(1 for n in nums if n % 5 == 0)
    divisible_7 = sum(1 for n in nums if n % 7 == 0)
    semiprime_count = sum(1 for n in nums if OMEGA_TABLE[n] == 2)
    return {
        "prime_count": prime_count,
        "composite_count": composite_count,
        "one_count": one_count,
        "omega_sum": omega_sum,
        "distinct_factor_sum": distinct_factor_sum,
        "squarefree_count": squarefree_count,
        "divisible_2": divisible_2,
        "divisible_3": divisible_3,
        "divisible_5": divisible_5,
        "divisible_7": divisible_7,
        "semiprime_count": semiprime_count,
    }


def _factor_component_stats(draws: list[Draw], recent_window: int = 60) -> dict[str, dict[int, float]]:
    """소인수 p별 장기/최근/갭 통계를 계산한다."""
    latest = draws[-1].round
    recent = draws[-recent_window:] if len(draws) >= recent_window else draws
    total_exp = {p: 0.0 for p in PRIME_BASIS}
    recent_exp = {p: 0.0 for p in PRIME_BASIS}
    present_rounds = {p: [] for p in PRIME_BASIS}

    for d in draws:
        draw_has = {p: False for p in PRIME_BASIS}
        for n in d.numbers:
            for p, e in FACTOR_TABLE[n].items():
                if e:
                    total_exp[p] += e
                    draw_has[p] = True
        for p, ok in draw_has.items():
            if ok:
                present_rounds[p].append(d.round)

    for d in recent:
        for n in d.numbers:
            for p, e in FACTOR_TABLE[n].items():
                if e:
                    recent_exp[p] += e

    gap_ratio = {}
    current_gap = {}
    avg_gap = {}
    for p, rounds in present_rounds.items():
        if len(rounds) <= 1:
            current_gap[p] = float(latest - rounds[-1]) if rounds else float(len(draws))
            avg_gap[p] = float(len(draws))
            gap_ratio[p] = 0.0
        else:
            gaps = [b - a for a, b in zip(rounds, rounds[1:])]
            cur = latest - rounds[-1]
            avg = sum(gaps) / len(gaps)
            current_gap[p] = float(cur)
            avg_gap[p] = float(avg)
            gap_ratio[p] = float(cur / (avg + 1e-9))

    return {
        "total_exp": total_exp,
        "recent_exp": recent_exp,
        "gap_ratio": gap_ratio,
        "current_gap": current_gap,
        "avg_gap": avg_gap,
    }


class PrimeFactorStrategy(BaseStrategy):
    name = "prime_factor"
    explanation = "당첨번호를 소인수분해하여 소인수 성분의 장기/최근/갭 신호를 점수화"

    def __init__(self, recent_window: int = 60):
        self.recent_window = recent_window

    def score(self, draws: list[Draw]) -> StrategyOutput:
        stats = _factor_component_stats(draws, self.recent_window)
        z_total = _z({p: stats["total_exp"][p] for p in PRIME_BASIS})
        z_recent = _z({p: stats["recent_exp"][p] for p in PRIME_BASIS})
        z_gap = _z({p: stats["gap_ratio"][p] for p in PRIME_BASIS})

        # 소인수별 가중치: 장기 30%, 최근 35%, overdue 25%, 안정성 10%.
        factor_weight = {
            p: 0.30 * z_total[p] + 0.35 * z_recent[p] + 0.25 * z_gap[p] + 0.10 / math.sqrt(p)
            for p in PRIME_BASIS
        }

        raw = {}
        for n in range(1, 46):
            expv = FACTOR_TABLE[n]
            omega = OMEGA_TABLE[n]
            if n == 1:
                # 1은 소인수가 없으므로 과도하게 불리하지 않게 약한 중립값을 준다.
                raw[n] = -0.15
                continue
            # 지수가 큰 합성수에 과도한 점수가 쌓이지 않도록 sqrt(omega)로 정규화.
            component = sum(min(e, 2) * factor_weight[p] for p, e in expv.items() if e > 0)
            component = component / math.sqrt(max(1, omega))
            # square-free와 semi-prime은 과거 조합에서 흔한 구조라 약한 보정만 부여.
            structure = 0.06 * SQUAREFREE_TABLE[n] + (0.04 if omega == 2 else 0.0)
            # 큰 소수는 사람 선택 편향을 덜 받는 경향이 있어 아주 약하게 반영.
            large_prime = 0.04 if n in PRIME_SET and n > 31 else 0.0
            raw[n] = component + structure + large_prime

        return StrategyOutput(self.name, _z(raw), self.explanation)


def prime_combo_reference(draws: list[Draw]) -> dict[str, tuple[float, float]]:
    """과거 draw들의 소인수 구조 평균/표준편차를 1회 계산해 캐싱용으로 반환한다."""
    profiles = [draw_prime_profile(d.numbers) for d in draws]
    keys = [
        "prime_count", "composite_count", "omega_sum", "distinct_factor_sum",
        "squarefree_count", "divisible_2", "divisible_3", "divisible_5",
        "divisible_7", "semiprime_count",
    ]
    ref: dict[str, tuple[float, float]] = {}
    for k in keys:
        arr = np.array([float(p[k]) for p in profiles], dtype=float)
        ref[k] = (float(arr.mean()), float(arr.std()) or 1.0)
    return ref


def prime_combo_structural_score(
    combo: list[int] | tuple[int, ...],
    draws: list[Draw] | None = None,
    ref: dict[str, tuple[float, float]] | None = None,
) -> float:
    """
    후보 조합의 소인수 구조가 과거 1~현재 회차의 흔한 소인수 구조와
    얼마나 가까운지 점수화한다. generate_candidates에서 보조점수로 사용한다.
    ref를 넘기면 후보 생성 시 반복 계산을 피할 수 있다.
    """
    if ref is None:
        if draws is None:
            raise ValueError("draws 또는 ref 중 하나가 필요합니다.")
        ref = prime_combo_reference(draws)
    target = draw_prime_profile(combo)
    score = 0.0
    for k, (mu, sd) in ref.items():
        z = abs((float(target[k]) - mu) / sd)
        score += max(-2.5, -z)
    return float(score / max(1, len(ref)))


# -----------------------------------------------------------------------------
# v7.0 추가 전략 (모두 train 구간 draws만 사용 → 미래 누수 없음)
# -----------------------------------------------------------------------------


class MarkovTransitionStrategy(BaseStrategy):
    name = "markov_transition"
    explanation = "직전 회차 번호 m → 다음 회차 번호 n 전이확률 P(n|m) (Laplace 평활)"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def score(self, draws: list[Draw]) -> StrategyOutput:
        trans = np.zeros((46, 46), dtype=float)
        occ = np.zeros(46, dtype=float)
        for prev, nxt in zip(draws, draws[1:]):
            for m in prev.numbers:
                occ[m] += 1
                for n in nxt.numbers:
                    trans[m, n] += 1
        base = 6 / 45
        last = draws[-1].numbers
        scores = {}
        for n in range(1, 46):
            # 직전 회차 각 번호 m에 대한 전이확률의 기저확률 대비 lift 평균
            lifts = [((trans[m, n] + self.alpha * base) / (occ[m] + self.alpha)) / base for m in last]
            scores[n] = float(np.mean(lifts))
        return StrategyOutput(self.name, _z(scores), self.explanation)


class MultiScaleFrequencyStrategy(BaseStrategy):
    name = "multi_scale_freq"
    explanation = "10/25/50/100회 다중 윈도우 Dirichlet 평활 빈도 평균"

    def __init__(self, windows: tuple[int, ...] = (10, 25, 50, 100), prior: float = 5.0):
        self.windows = windows
        self.prior = prior

    def score(self, draws: list[Draw]) -> StrategyOutput:
        acc = {n: 0.0 for n in range(1, 46)}
        for w in self.windows:
            recent = draws[-w:]
            c = Counter(n for d in recent for n in d.numbers)
            # prior 회차 상당의 균등 prior로 수축
            z = _z({n: (c[n] + self.prior * 6 / 45) / (len(recent) + self.prior) for n in range(1, 46)})
            for n in range(1, 46):
                acc[n] += z[n] / len(self.windows)
        return StrategyOutput(self.name, _z(acc), self.explanation)


class LastDigitStrategy(BaseStrategy):
    name = "last_digit"
    explanation = "끝수(0~9) 최근 50회 출현 비율 대비 기대치 lift를 같은 끝수 번호에 배분"

    def __init__(self, window: int = 50):
        self.window = window

    def score(self, draws: list[Draw]) -> StrategyOutput:
        recent = draws[-self.window:]
        c = Counter(n % 10 for d in recent for n in d.numbers)
        size = Counter(n % 10 for n in range(1, 46))
        total = 6 * len(recent)
        scores = {n: (c[n % 10] + 1) / (total * size[n % 10] / 45 + 1) for n in range(1, 46)}
        return StrategyOutput(self.name, _z(scores), self.explanation)


ZONES = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]


def zone_of(n: int) -> int:
    for i, (lo, hi) in enumerate(ZONES):
        if lo <= n <= hi:
            return i
    raise ValueError(n)


class ZoneBalanceStrategy(BaseStrategy):
    name = "zone_balance"
    explanation = "번호대(1-9/10대/20대/30대/40대) 최근 30회 과소출현 구간 평균회귀 보정"

    def __init__(self, window: int = 30):
        self.window = window

    def score(self, draws: list[Draw]) -> StrategyOutput:
        recent = draws[-self.window:]
        c = Counter(zone_of(n) for d in recent for n in d.numbers)
        total = 6 * len(recent)
        scores = {}
        for n in range(1, 46):
            zi = zone_of(n)
            lo, hi = ZONES[zi]
            expected = total * (hi - lo + 1) / 45
            scores[n] = (expected - c[zi]) / math.sqrt(expected + 1)
        return StrategyOutput(self.name, _z(scores), self.explanation)


class BonusCarryStrategy(BaseStrategy):
    name = "bonus_carry"
    explanation = "최근 보너스 번호의 본번호 전환 신호 (반감기 5회 감쇠)"

    def __init__(self, half_life: float = 5.0, window: int = 20):
        self.half_life = half_life
        self.window = window

    def score(self, draws: list[Draw]) -> StrategyOutput:
        scores = {n: 0.0 for n in range(1, 46)}
        recent = draws[-self.window:]
        for age, d in enumerate(reversed(recent)):
            scores[d.bonus] += 0.5 ** (age / self.half_life)
        return StrategyOutput(self.name, _z(scores), self.explanation)


def ac_value(nums: list[int] | tuple[int, ...]) -> int:
    """AC(Arithmetic Complexity) 값: 서로 다른 양의 차이 개수 - 5."""
    s = sorted(nums)
    diffs = {b - a for i, a in enumerate(s) for b in s[i + 1:]}
    return len(diffs) - (len(s) - 1)


def default_strategies() -> list[BaseStrategy]:
    return [
        LongFrequencyStrategy(),
        RecentDecayStrategy(half_life=45.0),
        GapStrategy(),
        HotColdStrategy(),
        MomentumStrategy(),
        BayesStrategy(),
        PairAffinityStrategy(),
        PrimeFactorStrategy(recent_window=60),
        AntiCrowdStrategy(),
        MarkovTransitionStrategy(),
        MultiScaleFrequencyStrategy(),
        LastDigitStrategy(),
        ZoneBalanceStrategy(),
        BonusCarryStrategy(),
    ]


def strategy_score_frame(outputs: list[StrategyOutput]) -> pd.DataFrame:
    rows = []
    for n in range(1, 46):
        row = {"number": n}
        for out in outputs:
            row[out.name] = out.scores[n]
        rows.append(row)
    return pd.DataFrame(rows)


def pair_matrix(draws: list[Draw]) -> dict[int, dict[int, float]]:
    mat = {i: {j: 0.0 for j in range(1, 46)} for i in range(1, 46)}
    for d in draws:
        nums = list(d.numbers)
        for idx, a in enumerate(nums):
            for b in nums[idx + 1:]:
                mat[a][b] += 1.0
                mat[b][a] += 1.0
    # shrinkage toward expected pair count
    expected = len(draws) * (6 / 45) * (5 / 44)
    for i in range(1, 46):
        for j in range(1, 46):
            if i != j:
                mat[i][j] = (mat[i][j] + 0.25 * expected) / 1.25
    return mat


def combine_scores(outputs: list[StrategyOutput], weights: dict[str, float] | None = None) -> dict[int, float]:
    if weights is None:
        weights = {o.name: 1.0 / len(outputs) for o in outputs}
    combined = {n: 0.0 for n in range(1, 46)}
    for out in outputs:
        w = float(weights.get(out.name, 0.0))
        for n in range(1, 46):
            combined[n] += w * out.scores[n]
    return _minmax_positive(combined)
