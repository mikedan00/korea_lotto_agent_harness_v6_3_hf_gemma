from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .data_loader import Draw, draws_to_frame
from .metrics import hit_count, combo_stats, exact_probability_summary, brier_score_for_number_probs
from .strategies import default_strategies, strategy_score_frame, combine_scores, pair_matrix, prime_combo_structural_score, prime_combo_reference, draw_prime_profile


def rolling_backtest(draws: list[Draw], cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = default_strategies()
    start_idx = max(cfg.min_train_rounds, len(draws) - cfg.backtest_window)
    detail_rows = []
    summary_hits: dict[str, list[int]] = defaultdict(list)
    summary_brier: dict[str, list[float]] = defaultdict(list)

    for idx in range(start_idx, len(draws)):
        train = draws[:idx]
        actual = draws[idx]
        outputs = [s.score(train) for s in strategies]
        for out in outputs:
            top6 = sorted(range(1, 46), key=lambda n: out.scores[n], reverse=True)[:6]
            hits = hit_count(top6, actual.numbers)
            # convert z scores to pseudo probabilities around base rate; for comparative Brier only.
            vals = np.array([out.scores[n] for n in range(1, 46)], dtype=float)
            expv = np.exp(vals - vals.max())
            probs_arr = expv / expv.sum() * 6.0
            probs = {n: float(min(1.0, max(0.0, probs_arr[n - 1]))) for n in range(1, 46)}
            brier = brier_score_for_number_probs(probs, actual.numbers)
            summary_hits[out.name].append(hits)
            summary_brier[out.name].append(brier)
            detail_rows.append({
                "round": actual.round,
                "strategy": out.name,
                "prediction": "-".join(map(str, sorted(top6))),
                "actual": "-".join(map(str, actual.numbers)),
                "hits": hits,
                "brier": brier,
            })

    summary_rows = []
    for name in summary_hits:
        hits = np.array(summary_hits[name], dtype=float)
        brier = np.array(summary_brier[name], dtype=float)
        summary_rows.append({
            "strategy": name,
            "folds": int(len(hits)),
            "mean_hits": float(hits.mean()) if len(hits) else 0.0,
            "hit_ge_2_rate": float((hits >= 2).mean()) if len(hits) else 0.0,
            "hit_ge_3_rate": float((hits >= 3).mean()) if len(hits) else 0.0,
            "mean_brier": float(brier.mean()) if len(brier) else 0.0,
        })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        composite = summary["mean_hits"] + 2.0 * summary["hit_ge_3_rate"] - 1.5 * summary["mean_brier"]
        vals = np.exp((composite - composite.max()) / 0.25)
        summary["softmax_weight"] = vals / vals.sum()
    return pd.DataFrame(detail_rows), summary.sort_values("softmax_weight", ascending=False) if not summary.empty else summary


def weights_from_summary(summary: pd.DataFrame) -> dict[str, float]:
    if summary.empty or "softmax_weight" not in summary.columns:
        names = [s.name for s in default_strategies()]
        return {n: 1 / len(names) for n in names}
    return {str(r["strategy"]): float(r["softmax_weight"]) for _, r in summary.iterrows()}


def generate_candidates(draws: list[Draw], number_scores: dict[int, float], weights: dict[str, float], cfg: AppConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    nums = list(range(1, 46))
    raw_weights = np.array([max(1e-6, number_scores[n]) for n in nums], dtype=float)
    probs = raw_weights / raw_weights.sum()
    pair = pair_matrix(draws)
    last = set(draws[-1].numbers)
    prime_ref = prime_combo_reference(draws)
    seen = set()
    rows = []

    def valid(combo: list[int]) -> bool:
        st = combo_stats(combo, last)
        if not (cfg.sum_min <= st["sum"] <= cfg.sum_max): return False
        if not (2 <= st["odd"] <= 4): return False
        if not (2 <= st["high"] <= 4): return False
        if st["consecutive_pairs"] > 2: return False
        if st["last_overlap"] > 2: return False
        return True

    max_iter = max(cfg.candidate_samples * 8, 5000)
    for _ in range(max_iter):
        if len(rows) >= cfg.candidate_samples:
            break
        combo = sorted(rng.choices(nums, weights=probs, k=12))
        # keep unique ordered first 6; if duplicate shortage, use weighted sample fallback.
        u = []
        for x in combo:
            if x not in u:
                u.append(x)
            if len(u) == 6:
                break
        if len(u) < 6:
            u = sorted(rng.sample(nums, 6))
        combo = sorted(u)
        key = tuple(combo)
        if key in seen or not valid(combo):
            continue
        seen.add(key)
        st = combo_stats(combo, last)
        st.pop("numbers", None)
        num_score = sum(number_scores[n] for n in combo)
        pair_sum = 0.0
        for i, a in enumerate(combo):
            for b in combo[i+1:]:
                pair_sum += pair[a][b]
        # structural prior around historical common region.
        structural = 0.0
        structural -= abs(st["sum"] - 138) / 80.0
        structural -= abs(st["odd"] - 3) * 0.05
        structural -= abs(st["high"] - 3) * 0.05
        structural -= st["consecutive_pairs"] * 0.03
        structural -= st["last_overlap"] * 0.05
        prime_struct = prime_combo_structural_score(combo, ref=prime_ref)
        prime_profile = draw_prime_profile(combo)
        score = num_score + 0.015 * pair_sum + structural + 0.35 * prime_struct
        rows.append({
            "numbers": ", ".join(map(str, combo)),
            "score": score,
            "number_score": num_score,
            "pair_score": pair_sum,
            "prime_struct_score": prime_struct,
            "prime_count": prime_profile["prime_count"],
            "omega_sum": prime_profile["omega_sum"],
            "squarefree_count": prime_profile["squarefree_count"],
            "semiprime_count": prime_profile["semiprime_count"],
            **st,
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def select_final_sets(candidate_df: pd.DataFrame, n_sets: int = 2, max_overlap: int = 2) -> list[list[int]]:
    selected: list[list[int]] = []
    for _, row in candidate_df.iterrows():
        nums = [int(x.strip()) for x in str(row["numbers"]).split(",")]
        if all(len(set(nums) & set(prev)) <= max_overlap for prev in selected):
            selected.append(nums)
        if len(selected) >= n_sets:
            break
    return selected


def run_core_pipeline(draws: list[Draw], cfg: AppConfig) -> dict[str, Any]:
    detail, summary = rolling_backtest(draws, cfg)
    weights = weights_from_summary(summary)
    outputs = [s.score(draws) for s in default_strategies()]
    number_scores = combine_scores(outputs, weights)
    score_df = strategy_score_frame(outputs)
    score_df["ensemble_score"] = score_df["number"].map(number_scores)
    candidate_df = generate_candidates(draws, number_scores, weights, cfg)
    final_sets = select_final_sets(candidate_df, cfg.n_sets, cfg.max_overlap)
    return {
        "history": draws_to_frame(draws),
        "backtest_detail": detail,
        "backtest_summary": summary,
        "weights": pd.DataFrame([{"strategy": k, "weight": v} for k, v in weights.items()]).sort_values("weight", ascending=False),
        "number_scores": score_df.sort_values("ensemble_score", ascending=False),
        "candidates": candidate_df,
        "final_sets": final_sets,
        "probability": exact_probability_summary(),
    }


def save_pipeline_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key in ["history", "backtest_detail", "backtest_summary", "weights", "number_scores", "candidates"]:
        obj = result.get(key)
        if isinstance(obj, pd.DataFrame):
            path = out / f"{key}.csv"
            obj.to_csv(path, index=False, encoding="utf-8-sig")
            paths[key] = str(path)
    serial = {
        "final_sets": result.get("final_sets", []),
        "probability": result.get("probability", {}),
        "top_candidates": result.get("candidates", pd.DataFrame()).head(20).to_dict(orient="records") if isinstance(result.get("candidates"), pd.DataFrame) else [],
    }
    json_path = out / "prediction_result.json"
    json_path.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["prediction_result"] = str(json_path)
    return paths
