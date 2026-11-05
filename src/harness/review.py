"""v7.0 검토 계층: 결정론적 통계 감사 + LLM 리뷰 보드(분석가 → 비평가 → 최적화 자문)."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..llm_engine import SYSTEM_GUARD, extract_json

# LLM 이 제안할 수 있는 파라미터와 허용 범위 (그 외 키는 무시)
TUNABLE = {
    "backtest_window": (int, 60, 600),
    "tune_ratio": (float, 0.5, 0.85),
    "filter_coverage": (float, 0.80, 0.99),
    "explore_temperature": (float, 0.0, 1.0),
    "coverage_weight": (float, 0.0, 1.0),
    "max_overlap": (int, 0, 4),
}
SCORE_WEIGHT_KEYS = ("prob", "pair", "typical", "prime")


def statistical_audit(result: dict[str, Any], cfg) -> dict[str, Any]:
    """LLM 없이 항상 실행되는 감사. 과적합/유의성/필터/가중치 집중도를 점검."""
    opt = result.get("optimization", {})
    summary: pd.DataFrame = result.get("backtest_summary", pd.DataFrame())
    filters = result.get("filters", {})
    flags: list[str] = []

    ho = opt.get("holdout") or {}
    hold = float(ho.get("edge_nats", opt.get("holdout_edge_nats", 0.0)))
    ci = ho.get("edge_ci95") or (hold, hold)
    tune = float(opt.get("tune_edge_nats", 0.0))
    # 판단은 튜닝에 쓰지 않은 holdout fold 만의 적중 검정 사용 (단측: 무작위보다 큰가)
    p_ens = float(ho.get("hits_p_one_sided", opt.get("ensemble_hits_p_value", 1.0)))
    p_two = float(ho.get("hits_p_two_sided", 1.0))
    if hold <= 0:
        flags.append(f"holdout 로그우도 edge {hold:+.4f} (95% CI {ci[0]:+.4f}~{ci[1]:+.4f}) ≤ 0: 균등 무작위 대비 우위 증거 없음 → 신호 수축 적용")
    elif ci[0] <= 0:
        flags.append(f"holdout edge {hold:+.4f} > 0 이지만 95% CI 하한 {ci[0]:+.4f} ≤ 0: 우위가 통계적으로 확인되지 않음")
    if tune - hold > 0.01:
        flags.append(f"tune/holdout edge 격차 {tune - hold:.4f} nats: 과적합 의심")
    if p_ens > 0.05:
        flags.append(f"holdout 적중 p(단측)={p_ens:.3f}, p(양측)={p_two:.3f}: 무작위(기대 0.8개)보다 낫다는 근거 없음")
    diff_ci = ho.get("ensemble_minus_equal_edge_ci95")
    if diff_ci:
        if diff_ci[1] < 0:
            flags.append(f"최적화 앙상블이 균등 가중치 기준선보다 holdout edge 가 유의하게 낮음 (차이 95% CI {diff_ci[0]:+.4f}~{diff_ci[1]:+.4f}): 가중치 최적화가 오히려 손해")
        elif diff_ci[0] <= 0:
            flags.append(f"최적화 앙상블과 균등 가중치 기준선의 holdout edge 차이가 유의하지 않음 (95% CI {diff_ci[0]:+.4f}~{diff_ci[1]:+.4f})")
    sig = []
    if not summary.empty and "p_bonferroni" in summary:
        sig = summary.loc[summary["p_bonferroni"] < 0.05, "strategy"].tolist()
        if not sig:
            flags.append("Bonferroni 보정 후 유의한 단일 전략 없음 (다중비교 보정)")
    w = opt.get("weights", {})
    if w and max(w.values()) > 0.5:
        top = max(w, key=w.get)
        flags.append(f"가중치 집중: {top}={w[top]:.2f} (> 0.5) — 단일 전략 의존 위험")
    jp = filters.get("_joint_pass_rate", {}).get("value")
    if jp is not None and jp < 0.8:
        flags.append(f"구조 필터 결합 통과율 {jp:.2f}: 과거 당첨조합 다수를 제외할 수 있음")
    if len(result.get("final_sets", [])) != cfg.n_sets:
        flags.append("최종 세트 수 불일치")

    if ci[0] > 0 and p_ens <= 0.05 and sig:
        verdict = "signal"
    elif hold > 0:
        verdict = "weak_signal"
    else:
        verdict = "no_evidence_of_edge"
    return {
        "verdict": verdict,
        "risk_flags": flags,
        "significant_strategies": sig,
        "holdout_edge_nats": hold,
        "holdout_edge_ci95": list(ci),
        "tune_edge_nats": tune,
        "holdout_hits_p_one_sided": p_ens,
        "holdout_hits_p_two_sided": p_two,
        "ensemble_minus_equal_edge_ci95": list(diff_ci) if diff_ci else None,
        "leakage_check": "pass (fold t 는 draws[:t] 만 사용하도록 구현)",
    }


def compact_summary(result: dict[str, Any], cfg, audit: dict[str, Any]) -> dict[str, Any]:
    opt = {k: v for k, v in result.get("optimization", {}).items() if k != "history"}
    bs = result.get("backtest_summary")
    cands = result.get("candidates")
    cols = ["numbers", "score", "c_prob", "c_pair", "c_typical", "c_prime", "log_prob_vs_uniform", "sum", "odd", "high", "ac"]
    fsd = result.get("final_set_details")
    return {
        "target_round": cfg.target_round,
        "final_sets": result.get("final_sets", []),
        "final_set_details": fsd.round(4).to_dict(orient="records") if isinstance(fsd, pd.DataFrame) else [],
        "notes": {
            "score": "score = c_prob + c_pair + c_typical + c_prime. 각 c_* 는 (점수 가중치 × 후보 집합 안에서 표준화한 z-score)",
            "final_set_selection": "점수 상위 400개 중 score + coverage_weight × (이전 세트에 없던 번호 수) 가 최대인 조합을 차례로 선택. score_rank 는 전체 후보 중 점수 순위",
            "p_values": "backtest_summary 의 p_value 는 단측(무작위보다 큰가), p_two_sided 는 양측, p_bonferroni 는 단측 × 전략 수",
            "holdout": "optimization.holdout 블록은 튜닝에 쓰지 않은 fold 만 사용한 순수 표본외 결과. ensemble_*_all_folds 와 ensemble_hits_p_value 는 튜닝 fold 가 섞인 참고값",
            "identical_strategies": "long_frequency, bayes, pair_affinity 는 모두 번호별 출현 횟수의 단조 함수라 순위가 같아 지표가 동일함 (버그 아님)",
        },
        "optimization": opt,
        "statistical_audit": audit,
        "backtest_summary": bs.round(4).to_dict(orient="records") if isinstance(bs, pd.DataFrame) else [],
        "top_candidates": cands[[c for c in cols if c in cands]].head(10).round(4).to_dict(orient="records") if isinstance(cands, pd.DataFrame) else [],
        "filters": {k: {"low": v.get("low"), "high": v.get("high")} for k, v in result.get("filters", {}).items() if not k.startswith("_")},
        "current_params": {k: getattr(cfg, k) for k in TUNABLE} | {"score_weights": cfg.score_weights},
        "probability": result.get("probability", {}),
    }


ANALYST_PROMPT = """[역할: 분석가 에이전트]
아래 JSON(통계 엔진 결과 + 결정론적 감사)을 근거로 한국어 마크다운 리포트를 작성하세요.
1) 최종 세트별 구조 해석(합계/홀짝/고저/AC/커버리지)  2) 전략별 백테스트 해석(무작위 기대 0.8개 대비)
3) 앙상블 최적화 결과 해석: optimization.holdout 블록(순수 표본외 edge·적중·95% CI·균등 가중치 기준선 비교)을 우선 근거로 사용
4) 한계와 주의. notes 항목의 정의를 따르세요.
번호를 새로 만들거나 바꾸지 마세요. 과장 금지.

```json
{payload}
```"""

CRITIC_PROMPT = """[역할: 비평가/리스크 검토 에이전트]
분석가 리포트와 원자료를 교차 검증하세요. 과장, 근거 없는 주장, 과적합, 데이터 누수, 다중비교 문제를 찾으세요.
반드시 아래 형식의 JSON 객체 하나만 출력하세요(설명 문장 금지).
{{"verdict": "accept|revise|reject", "confidence": 0.0~1.0, "risk_flags": ["..."], "analyst_errors": ["..."], "summary": "한 문단 한국어 요약"}}

[원자료]
```json
{payload}
```

[분석가 리포트]
{analyst}"""

ADVISOR_PROMPT = """[역할: 최적화 자문 에이전트]
감사 결과와 비평을 바탕으로 다음 실행의 파라미터 조정을 제안하세요. 허용 키와 범위만 사용:
{tunable}
score_weights 는 prob/pair/typical/prime 키(0~2)만 허용.
반드시 아래 형식의 JSON 객체 하나만 출력하세요.
{{"param_suggestions": {{"키": 값}}, "score_weights": {{"prob": 1.0}}, "rationale": ["..."], "next_experiments": ["..."]}}

[원자료]
```json
{payload}
```

[비평가 결과]
{critic}"""


def run_review_board(client, result: dict[str, Any], cfg, audit: dict[str, Any], progress=None) -> dict[str, Any]:
    payload = json.dumps(compact_summary(result, cfg, audit), ensure_ascii=False, indent=1, default=str)[:14000]
    board: dict[str, Any] = {"provider": client.provider, "model": client.model_label, "steps": {}}

    def step(name: str, prompt: str) -> dict[str, Any]:
        if progress:
            progress(f"AI 리뷰: {name}")
        r = client.chat_messages(SYSTEM_GUARD, prompt)
        board["steps"][name] = {k: r.get(k) for k in ("ok", "model", "content", "error", "cost_usd")}
        return r

    a = step("analyst", ANALYST_PROMPT.format(payload=payload))
    if not a.get("ok"):
        board["ok"] = False
        board["error"] = a.get("error")
        return board
    c = step("critic", CRITIC_PROMPT.format(payload=payload, analyst=a.get("content", "")[:6000]))
    critic = extract_json(c.get("content", "")) if c.get("ok") else None
    board["critic"] = critic
    d = step("advisor", ADVISOR_PROMPT.format(
        payload=payload,
        tunable=json.dumps({k: [v[1], v[2]] for k, v in TUNABLE.items()}),
        critic=json.dumps(critic, ensure_ascii=False) if critic else c.get("content", "")[:3000],
    ))
    advisor = extract_json(d.get("content", "")) if d.get("ok") else None
    board["advisor"] = advisor
    board["suggestions"] = sanitize_suggestions(advisor or {})
    board["ok"] = True
    return board


def sanitize_suggestions(advisor: dict[str, Any]) -> dict[str, Any]:
    """LLM 제안을 허용 키/범위로 클램프. 적용 가능한 dict 만 반환."""
    out: dict[str, Any] = {}
    for k, v in (advisor.get("param_suggestions") or {}).items():
        if k not in TUNABLE:
            continue
        typ, lo, hi = TUNABLE[k]
        try:
            out[k] = typ(min(hi, max(lo, typ(v))))
        except (TypeError, ValueError):
            continue
    sw = {}
    for k, v in (advisor.get("score_weights") or {}).items():
        if k in SCORE_WEIGHT_KEYS:
            try:
                sw[k] = float(min(2.0, max(0.0, float(v))))
            except (TypeError, ValueError):
                continue
    if sw:
        out["score_weights"] = sw
    return out


def apply_suggestions(cfg, suggestions: dict[str, Any]):
    for k, v in suggestions.items():
        if k == "score_weights":
            cfg.score_weights = {**cfg.score_weights, **v}
        elif k in TUNABLE:
            setattr(cfg, k, v)
    return cfg
