from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .data_loader import Draw, draws_to_frame
from .metrics import exact_probability_summary
from .optimizer import (
    EXPECTED_HITS,
    FoldCache,
    collect_fold_scores,
    ensemble_logits,
    ensemble_probabilities,
    hits_significance,
    optimize_ensemble,
    strategy_report,
    top6_hits,
)
from .strategies import (
    PRIME_SET,
    OMEGA_TABLE,
    SQUAREFREE_TABLE,
    default_strategies,
    pair_matrix,
    strategy_score_frame,
    zone_of,
)

STRUCT_METRICS = ["sum", "odd", "high", "consecutive_pairs", "last_overlap", "ac", "zones_used", "max_same_last_digit", "prime_count"]


# -----------------------------------------------------------------------------
# 1. walk-forward 백테스트 + 앙상블 최적화
# -----------------------------------------------------------------------------

def backtest_start_index(draws: list[Draw], cfg: AppConfig) -> int:
    return max(cfg.min_train_rounds, len(draws) - cfg.backtest_window)


def build_backtest_tables(cache: FoldCache) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for i, name in enumerate(cache.names):
        top = np.argsort(-cache.scores[:, i, :], axis=1)[:, :6] + 1
        for f, rnd in enumerate(cache.rounds):
            actual = np.flatnonzero(cache.actual[f]) + 1
            pred = sorted(int(x) for x in top[f])
            detail_rows.append({
                "round": rnd,
                "strategy": name,
                "prediction": "-".join(map(str, pred)),
                "actual": "-".join(map(str, actual)),
                "hits": len(set(pred) & set(actual.tolist())),
            })
    summary = pd.DataFrame(strategy_report(cache)).sort_values("ll_edge", ascending=False).reset_index(drop=True)
    return pd.DataFrame(detail_rows), summary


def rolling_backtest(draws: list[Draw], cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v6 호환 API: (detail, summary) 반환."""
    cache = collect_fold_scores(draws, default_strategies(), backtest_start_index(draws, cfg))
    return build_backtest_tables(cache)


# -----------------------------------------------------------------------------
# 2. 과거 데이터에서 학습한 구조 필터
# -----------------------------------------------------------------------------

def combo_metrics_matrix(combos: np.ndarray, last_numbers: tuple[int, ...] | list[int]) -> dict[str, np.ndarray]:
    """combos: (M,6) 정렬된 정수 배열 → 구조 지표 벡터들."""
    c = np.sort(combos, axis=1)
    m = {
        "sum": c.sum(1),
        "odd": (c % 2).sum(1),
        "high": (c >= 23).sum(1),
        "consecutive_pairs": (np.diff(c, axis=1) == 1).sum(1),
        "last_overlap": np.isin(c, np.asarray(list(last_numbers))).sum(1),
    }
    ii, jj = np.triu_indices(6, 1)
    diffs = np.sort(c[:, jj] - c[:, ii], axis=1)
    m["ac"] = (np.diff(diffs, axis=1) != 0).sum(1) + 1 - 5
    zone_lut = np.array([0] + [zone_of(n) for n in range(1, 46)])
    z = np.sort(zone_lut[c], axis=1)
    m["zones_used"] = (np.diff(z, axis=1) != 0).sum(1) + 1
    ld = c % 10
    m["max_same_last_digit"] = np.max(np.stack([(ld == d).sum(1) for d in range(10)], axis=1), axis=1)
    prime_lut = np.array([0] + [int(n in PRIME_SET) for n in range(1, 46)])
    m["prime_count"] = prime_lut[c].sum(1)
    return m


def history_metrics(draws: list[Draw]) -> dict[str, np.ndarray]:
    rows = {k: [] for k in STRUCT_METRICS}
    for prev, d in zip(draws, draws[1:]):
        mm = combo_metrics_matrix(np.array([d.numbers]), prev.numbers)
        for k in STRUCT_METRICS:
            rows[k].append(int(mm[k][0]))
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


def learn_filters(draws: list[Draw], coverage: float = 0.95) -> dict[str, dict[str, float]]:
    """과거 당첨조합의 coverage(기본 95%)를 포함하는 지표 범위 + 평균/표준편차."""
    hm = history_metrics(draws)
    a = (1 - coverage) / 2
    out = {}
    for k, v in hm.items():
        out[k] = {
            "low": float(np.quantile(v, a, method="lower")),
            "high": float(np.quantile(v, 1 - a, method="higher")),
            "mean": float(v.mean()),
            "std": float(v.std()) or 1.0,
            "pass_rate": 0.0,
        }
    # 필터 전체를 동시에 적용했을 때 과거 당첨조합 통과율(진단용)
    mask = np.ones(len(next(iter(hm.values()))), dtype=bool)
    for k, v in hm.items():
        mask &= (v >= out[k]["low"]) & (v <= out[k]["high"])
        out[k]["pass_rate"] = float(((v >= out[k]["low"]) & (v <= out[k]["high"])).mean())
    out["_joint_pass_rate"] = {"value": float(mask.mean())}
    return out


# -----------------------------------------------------------------------------
# 3. 후보 생성 (Gumbel top-k 샘플링 + 다중 성분 점수)
# -----------------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def _prime_struct_vec(c: np.ndarray, draws: list[Draw]) -> np.ndarray:
    """소인수 구조 전형성: omega_sum / squarefree / semiprime 의 과거 분포 대비 |z| 평균의 음수."""
    omega = np.array([0] + [OMEGA_TABLE[n] for n in range(1, 46)])
    sqf = np.array([0] + [SQUAREFREE_TABLE[n] for n in range(1, 46)])
    semi = np.array([0] + [int(OMEGA_TABLE[n] == 2) for n in range(1, 46)])
    hist = np.array([d.numbers for d in draws])
    score = np.zeros(len(c))
    for lut in (omega, sqf, semi):
        h = lut[hist].sum(1)
        mu, sd = h.mean(), h.std() or 1.0
        score -= np.abs((lut[c].sum(1) - mu) / sd)
    return score / 3


def generate_candidates(draws: list[Draw], probs: dict[int, float], ens_z: dict[int, float], cfg: AppConfig,
                        filters: dict[str, dict[str, float]] | None = None, prob_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    last = draws[-1].numbers
    filters = filters or learn_filters(draws, cfg.filter_coverage)

    # 탐색용 샘플링 분포: 앙상블 z 에 고정 탐색온도 적용 (확률 q 가 균등에 가까워도 상위 번호 탐색 유지)
    z = np.array([ens_z[n] for n in range(1, 46)])
    logp = cfg.explore_temperature * z
    m = int(cfg.candidate_samples * 4)
    keys = logp[None, :] + rng.gumbel(size=(m, 45))
    combos = np.sort(np.argpartition(-keys, 6, axis=1)[:, :6] + 1, axis=1)
    combos = np.unique(combos, axis=0)

    met = combo_metrics_matrix(combos, last)
    if cfg.use_learned_filters:
        mask = np.ones(len(combos), dtype=bool)
        for k in STRUCT_METRICS:
            mask &= (met[k] >= filters[k]["low"]) & (met[k] <= filters[k]["high"])
    else:
        mask = ((met["sum"] >= cfg.sum_min) & (met["sum"] <= cfg.sum_max) & (met["odd"] >= 2) & (met["odd"] <= 4)
                & (met["high"] >= 2) & (met["high"] <= 4) & (met["consecutive_pairs"] <= 2) & (met["last_overlap"] <= 2))
    combos = combos[mask]
    met = {k: v[mask] for k, v in met.items()}
    if len(combos) > cfg.candidate_samples:
        keep = rng.choice(len(combos), cfg.candidate_samples, replace=False)
        combos = combos[keep]
        met = {k: v[keep] for k, v in met.items()}

    q = np.array([0.0] + [probs[n] for n in range(1, 46)])
    logq = np.log(q[combos] + 1e-15).sum(1)

    pm = pair_matrix(draws)
    expected = max(1e-9, len(draws) * (6 / 45) * (5 / 44))
    lift = np.zeros((46, 46))
    for a in range(1, 46):
        for b in range(1, 46):
            if a != b:
                lift[a, b] = np.log((pm[a][b] + 1.0) / (expected + 1.0))
    ii, jj = np.triu_indices(6, 1)
    pair_score = lift[combos[:, ii], combos[:, jj]].sum(1)

    typical = np.zeros(len(combos))
    for k in STRUCT_METRICS:
        typical -= np.abs((met[k] - filters[k]["mean"]) / filters[k]["std"])
    typical /= len(STRUCT_METRICS)
    prime_struct = _prime_struct_vec(combos, draws)

    w = cfg.score_weights
    # prob_scale<1: holdout 에서 우위가 검증되지 않아 수축된 경우 확률 성분 비중도 함께 낮춤
    score = (prob_scale * w["prob"] * _zscore(logq) + w["pair"] * _zscore(pair_score)
             + w["typical"] * _zscore(typical) + w["prime"] * _zscore(prime_struct))

    df = pd.DataFrame({
        "numbers": [", ".join(map(str, r)) for r in combos],
        "score": score,
        "log_prob": logq,
        "log_prob_vs_uniform": logq - 6 * np.log(1 / 45),
        "pair_score": pair_score,
        "typicality": typical,
        "prime_struct_score": prime_struct,
        **{k: met[k].astype(int) for k in STRUCT_METRICS},
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def select_final_sets(candidate_df: pd.DataFrame, n_sets: int = 2, max_overlap: int = 2,
                      coverage_weight: float = 0.3, pool: int = 400) -> list[list[int]]:
    """점수 + 신규 번호 커버리지 보너스를 동시에 고려하는 greedy 선택 (세트 간 overlap 제한)."""
    top = candidate_df.head(pool)
    rows = [([int(x) for x in str(r).split(",")], float(s)) for r, s in zip(top["numbers"], top["score"])]
    selected: list[list[int]] = []
    covered: set[int] = set()
    while len(selected) < n_sets:
        best, best_val = None, -1e18
        for nums, s in rows:
            if nums in selected or any(len(set(nums) & set(p)) > max_overlap for p in selected):
                continue
            val = s + coverage_weight * len(set(nums) - covered)
            if val > best_val:
                best, best_val = nums, val
        if best is None:
            break
        selected.append(best)
        covered |= set(best)
    return selected


# -----------------------------------------------------------------------------
# 4. 파이프라인
# -----------------------------------------------------------------------------

def run_core_pipeline(draws: list[Draw], cfg: AppConfig, progress=None) -> dict[str, Any]:
    def tick(msg: str) -> None:
        if progress:
            progress(msg)

    strategies = default_strategies()
    tick("walk-forward 전략 점수 캐싱")
    cache = collect_fold_scores(draws, strategies, backtest_start_index(draws, cfg))
    detail, summary = build_backtest_tables(cache)

    tick("앙상블 가중치/온도 최적화")
    if cfg.optimize and len(cache.rounds) >= 10:
        fit = optimize_ensemble(cache, tune_ratio=cfg.tune_ratio, n_random=cfg.opt_random, n_refine=cfg.opt_refine, seed=cfg.seed)
    else:
        k = len(cache.names)
        fit = optimize_ensemble(cache, tune_ratio=cfg.tune_ratio, n_random=0, n_refine=0, seed=cfg.seed, betas=(0.1,))
        fit.weights = {n: 1.0 / k for n in cache.names}

    w_vec = np.array([fit.weights[n] for n in cache.names])
    ens_hits = top6_hits(ensemble_logits(cache.scores, w_vec, max(fit.beta, 1e-6)), cache.actual)
    ens_sig = hits_significance(ens_hits)
    cut = max(1, int(len(cache.rounds) * cfg.tune_ratio))

    tick("전체 이력 기준 번호 확률 산출")
    outputs = [s.score(draws) for s in strategies]
    probs = ensemble_probabilities(outputs, fit.weights, fit.beta)
    ens_raw = {n: sum(fit.weights[o.name] * o.scores[n] for o in outputs) for n in range(1, 46)}
    vals = np.array(list(ens_raw.values()))
    ens_z = {n: float((ens_raw[n] - vals.mean()) / (vals.std() or 1.0)) for n in range(1, 46)}

    score_df = strategy_score_frame(outputs)
    score_df["ensemble_z"] = score_df["number"].map(ens_z)
    score_df["prob"] = score_df["number"].map(probs)
    score_df["prob_vs_uniform"] = score_df["prob"] * 45

    tick("구조 필터 학습 및 후보 생성")
    filters = learn_filters(draws, cfg.filter_coverage)
    candidate_df = generate_candidates(draws, probs, ens_z, cfg, filters, prob_scale=0.5 if fit.shrunk else 1.0)
    final_sets = select_final_sets(candidate_df, cfg.n_sets, cfg.max_overlap, cfg.coverage_weight)

    optimization = {
        "weights": fit.weights,
        "beta": fit.beta,
        "tune_folds": cut,
        "holdout_folds": len(cache.rounds) - cut,
        "tune_edge_nats": fit.tune_edge,
        "holdout_edge_nats": fit.holdout_edge,
        "holdout_mean_hits": fit.holdout_hits,
        "random_expected_hits": EXPECTED_HITS,
        "ensemble_mean_hits_all_folds": float(ens_hits.mean()) if len(ens_hits) else 0.0,
        "ensemble_hits_z": ens_sig["z"],
        "ensemble_hits_p_value": ens_sig["p_value"],
        "shrunk_to_uniform": fit.shrunk,
        "history": fit.history,
    }
    return {
        "history": draws_to_frame(draws),
        "backtest_detail": detail,
        "backtest_summary": summary,
        "weights": pd.DataFrame([{"strategy": k, "weight": v} for k, v in fit.weights.items()]).sort_values("weight", ascending=False),
        "number_scores": score_df.sort_values("ensemble_z", ascending=False),
        "candidates": candidate_df,
        "final_sets": final_sets,
        "filters": filters,
        "optimization": optimization,
        "probability": exact_probability_summary(),
    }


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def save_pipeline_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key in ["history", "backtest_detail", "backtest_summary", "weights", "number_scores", "candidates"]:
        obj = result.get(key)
        if isinstance(obj, pd.DataFrame):
            path = out / f"{key}.csv"
            obj.to_csv(path, index=False, encoding="utf-8-sig")
            paths[key] = str(path)
    cands = result.get("candidates")
    serial = {
        "final_sets": result.get("final_sets", []),
        "optimization": result.get("optimization", {}),
        "filters": result.get("filters", {}),
        "probability": result.get("probability", {}),
        "top_candidates": cands.head(20).to_dict(orient="records") if isinstance(cands, pd.DataFrame) else [],
    }
    json_path = out / "prediction_result.json"
    json_path.write_text(json.dumps(_jsonable(serial), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["prediction_result"] = str(json_path)
    return paths
