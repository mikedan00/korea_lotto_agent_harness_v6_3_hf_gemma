"""v7.0 walk-forward 앙상블 최적화기.

핵심 아이디어
-------------
1. 백테스트 구간의 각 fold t 에 대해 draws[:t] 만으로 전략 점수를 계산하고(미래 누수 금지)
   (fold, strategy, 45) 텐서로 캐싱한다. 이후 가중치 탐색은 재계산 없이 빠르게 수행된다.
2. 앙상블 확률 q = softmax(beta * Σ_k w_k z_k) 를 정의하고,
   적절한 점수규칙(proper scoring rule)인 로그우도 Σ_{n∈actual} log q_n 을 최대화한다.
   균등 기준선은 6·log(1/45) 이며, 그 차이를 edge(nats/회차)로 보고한다.
3. 앞쪽 fold(tune)로 가중치/온도(beta)를 탐색하고, 뒤쪽 fold(holdout)에서 검증한다.
   holdout edge 가 0 이하이면 beta 를 수축하여 균등 분포에 가깝게 만든다(과적합 방어).
4. 전략별 top-6 적중 수를 초기하분포 기대값 0.8 과 비교하는 z-검정 + Bonferroni 보정을 제공한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data_loader import Draw
from .strategies import BaseStrategy

N = 45
UNIFORM_LL = 6 * math.log(1 / N)
EXPECTED_HITS = 36 / 45
# 초기하분포 분산: n*K/N*(N-K)/N*(N-n)/(N-1)
HITS_VAR = 6 * (6 / 45) * (39 / 45) * (39 / 44)


@dataclass
class FoldCache:
    rounds: list[int]
    names: list[str]
    scores: np.ndarray  # (F, K, 45) z-score
    actual: np.ndarray  # (F, 45) 0/1


@dataclass
class EnsembleFit:
    weights: dict[str, float]
    beta: float
    tune_edge: float
    holdout_edge: float
    holdout_hits: float
    shrunk: bool
    history: list[dict[str, Any]] = field(default_factory=list)


def collect_fold_scores(draws: list[Draw], strategies: list[BaseStrategy], start_idx: int) -> FoldCache:
    rounds, scores, actual = [], [], []
    for idx in range(start_idx, len(draws)):
        train = draws[:idx]
        outs = [s.score(train) for s in strategies]
        scores.append([[o.scores[n] for n in range(1, 46)] for o in outs])
        y = np.zeros(N)
        for n in draws[idx].numbers:
            y[n - 1] = 1.0
        actual.append(y)
        rounds.append(draws[idx].round)
    return FoldCache(rounds, [s.name for s in strategies], np.asarray(scores, dtype=float), np.asarray(actual, dtype=float))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def ensemble_logits(scores: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    # scores (F,K,45), w (K,) -> (F,45)
    return beta * np.einsum("fkn,k->fn", scores, w)


def mean_edge(scores: np.ndarray, actual: np.ndarray, w: np.ndarray, beta: float) -> float:
    if len(actual) == 0:
        return 0.0
    lp = _log_softmax(ensemble_logits(scores, w, beta))
    ll = (lp * actual).sum(axis=1)
    return float(ll.mean() - UNIFORM_LL)


def top6_hits(logits: np.ndarray, actual: np.ndarray) -> np.ndarray:
    top = np.argsort(-logits, axis=1)[:, :6]
    return np.take_along_axis(actual, top, axis=1).sum(axis=1)


def hits_significance(hits: np.ndarray) -> dict[str, float]:
    f = len(hits)
    if f == 0:
        return {"z": 0.0, "p_value": 1.0}
    z = (float(hits.mean()) - EXPECTED_HITS) / math.sqrt(HITS_VAR / f)
    p = 0.5 * math.erfc(z / math.sqrt(2))  # 단측: 기준보다 큰가?
    return {"z": z, "p_value": p}


def strategy_report(cache: FoldCache) -> list[dict[str, Any]]:
    rows = []
    k = len(cache.names)
    for i, name in enumerate(cache.names):
        s = cache.scores[:, i, :]
        hits = top6_hits(s, cache.actual)
        w = np.zeros(k)
        w[i] = 1.0
        # 전략 단독으로 신호를 켰을 때(beta>0)의 최선 edge. 음수면 균등보다 나쁨.
        best_edge = max(mean_edge(cache.scores, cache.actual, w, b) for b in (0.02, 0.05, 0.1, 0.2, 0.35))
        sig = hits_significance(hits)
        rows.append({
            "strategy": name,
            "folds": int(len(hits)),
            "mean_hits": float(hits.mean()) if len(hits) else 0.0,
            "hits_vs_random": float(hits.mean() - EXPECTED_HITS) if len(hits) else 0.0,
            "hit_ge_2_rate": float((hits >= 2).mean()) if len(hits) else 0.0,
            "hit_ge_3_rate": float((hits >= 3).mean()) if len(hits) else 0.0,
            "ll_edge": best_edge,
            "z_score": sig["z"],
            "p_value": sig["p_value"],
            "p_bonferroni": min(1.0, sig["p_value"] * k),
        })
    return rows


def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def optimize_ensemble(
    cache: FoldCache,
    tune_ratio: float = 0.7,
    n_random: int = 300,
    n_refine: int = 200,
    seed: int = 7,
    l2: float = 0.01,
    betas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45),
) -> EnsembleFit:
    rng = np.random.default_rng(seed)
    f = len(cache.rounds)
    k = len(cache.names)
    cut = max(1, int(f * tune_ratio)) if f > 1 else f
    s_tune, y_tune = cache.scores[:cut], cache.actual[:cut]
    s_hold, y_hold = cache.scores[cut:], cache.actual[cut:]

    uni = np.full(k, 1.0 / k)

    def obj(w: np.ndarray, b: float) -> float:
        # L2 정규화: 균등 가중치에서 멀어질수록 페널티 → 단일 전략 과의존 억제
        return mean_edge(s_tune, y_tune, w, b) - l2 * float(((w - uni) ** 2).sum())

    best_w = np.full(k, 1.0 / k)
    best_b, best = 0.0, 0.0  # beta=0 (균등) 은 edge 0
    history = []
    candidates = [np.full(k, 1.0 / k)] + [np.eye(k)[i] for i in range(k)]
    candidates += list(rng.dirichlet(np.full(k, 0.6), size=n_random))
    for w in candidates:
        for b in betas:
            v = obj(w, b)
            if v > best:
                best, best_w, best_b = v, w.copy(), b
    history.append({"stage": "random_search", "tune_edge": best, "beta": best_b})

    # 국소 개선: 가우시안 섭동 + 단체 투영 (hill climbing)
    step = 0.15
    for it in range(n_refine):
        w = _project_simplex(best_w + rng.normal(0, step, size=k))
        b = float(np.clip(best_b * math.exp(rng.normal(0, 0.15)), 0.0, 1.0))
        v = obj(w, b)
        if v > best:
            best, best_w, best_b = v, w, b
        if it % 50 == 49:
            step *= 0.6
    history.append({"stage": "hill_climb", "tune_edge": best, "beta": best_b})

    hold_edge = mean_edge(s_hold, y_hold, best_w, best_b) if len(y_hold) else 0.0
    hold_hits = float(top6_hits(ensemble_logits(s_hold, best_w, best_b), y_hold).mean()) if len(y_hold) else 0.0
    shrunk = False
    final_b = best_b
    if hold_edge <= 0:
        # holdout 에서 균등보다 못하면 신호를 크게 약화 (과적합 방어)
        final_b = best_b * 0.25
        shrunk = True
    return EnsembleFit(
        weights={n: float(v) for n, v in zip(cache.names, best_w)},
        beta=float(final_b),
        tune_edge=float(mean_edge(s_tune, y_tune, best_w, best_b)),
        holdout_edge=float(hold_edge),
        holdout_hits=hold_hits,
        shrunk=shrunk,
        history=history,
    )


def ensemble_probabilities(outputs, weights: dict[str, float], beta: float) -> dict[int, float]:
    """전체 이력 기준 전략 출력 → 번호별 확률 q_n (합=1)."""
    logits = np.zeros(N)
    for o in outputs:
        w = weights.get(o.name, 0.0)
        logits += w * np.array([o.scores[n] for n in range(1, 46)])
    lp = _log_softmax(beta * logits)
    q = np.exp(lp)
    return {n: float(q[n - 1]) for n in range(1, 46)}
