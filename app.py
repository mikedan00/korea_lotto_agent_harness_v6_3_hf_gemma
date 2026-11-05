from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness, agent_team_spec
from src.harness.review import apply_suggestions
from src.llm_engine import (
    CLAUDE_MODEL_PRESETS,
    HF_MODEL_PRESETS,
    PROVIDERS,
    ClaudeCliClient,
    HfRouterClient,
    detect_claude_cli,
)

st.set_page_config(page_title="Korea Lotto Agent Harness v7.0", layout="wide")
st.title("🎯 Korea Lotto 6/45 Agent Harness v7.0")
st.caption("14개 전략 walk-forward 백테스트 · 앙상블 최적화(tune/holdout) · 통계 감사 · AI 리뷰 보드 (HF Router / Claude Code CLI)")

CUSTOM = "✏️ 직접 입력"


@st.cache_data(show_spinner=False, ttl=300)
def _claude_detect(path: str) -> dict:
    return detect_claude_cli(path)


def _save_upload(upload, suffix: str) -> str | None:
    if upload is None:
        return None
    p = Path(tempfile.gettempdir()) / f"lotto_upload_{suffix}"
    p.write_bytes(upload.getvalue())
    return str(p)


def _model_picker(label: str, presets: list[str], default: str, key: str) -> str:
    options = list(dict.fromkeys(([default] if default else []) + presets)) + [CUSTOM]
    choice = st.selectbox(label, options, index=0, key=f"{key}_sel")
    if choice == CUSTOM:
        return st.text_input(f"{label} (직접 입력)", value=default, key=f"{key}_custom").strip()
    return choice


env_cfg = AppConfig.from_env()

with st.sidebar:
    st.header("① 실행 설정")
    target = st.number_input("목표 회차", min_value=1, value=1226, step=1)
    n_sets = st.number_input("최종 세트 수", min_value=1, max_value=10, value=2, step=1)
    samples = st.number_input("후보 조합 샘플 수", min_value=1000, max_value=200000, value=12000, step=1000)
    backtest = st.number_input("walk-forward 백테스트 회차", min_value=30, max_value=1000, value=180, step=10)
    max_overlap = st.number_input("세트 간 최대 중복", min_value=0, max_value=5, value=2, step=1)

    with st.expander("② 최적화 고급 설정"):
        optimize = st.checkbox("앙상블 가중치·온도 최적화", value=True)
        tune_ratio = st.slider("tune 비율 (나머지는 holdout 검증)", 0.5, 0.85, 0.7, 0.05)
        filter_coverage = st.slider("학습형 구조 필터 coverage", 0.80, 0.99, 0.95, 0.01)
        explore_temperature = st.slider("후보 탐색 온도", 0.0, 1.0, 0.35, 0.05)
        coverage_weight = st.slider("세트 간 커버리지 보너스", 0.0, 1.0, 0.3, 0.05)
        st.caption("후보 점수 가중치")
        sw = {
            "prob": st.slider("앙상블 확률", 0.0, 2.0, 1.0, 0.05),
            "pair": st.slider("pair lift", 0.0, 2.0, 0.25, 0.05),
            "typical": st.slider("구조 전형성", 0.0, 2.0, 0.35, 0.05),
            "prime": st.slider("소인수 구조", 0.0, 2.0, 0.15, 0.05),
        }

    st.subheader("③ 데이터")
    csv_file = st.file_uploader("기본 CSV 업로드", type=["csv"])
    xls_file = st.file_uploader("추가 회차 XLS/HTML 업로드", type=["xls", "html"])

    st.divider()
    st.header("④ AI 리뷰 모델")
    claude_info = _claude_detect(env_cfg.claude_cli_path)
    provider_keys = ["hf_api", "claude_cli", "none"]
    default_provider = env_cfg.llm_provider if env_cfg.llm_provider in provider_keys else "hf_api"
    provider = st.radio(
        "LLM 공급자", provider_keys, index=provider_keys.index(default_provider),
        format_func=lambda k: PROVIDERS[k] + ("" if k != "claude_cli" or claude_info["found"] else " — 미감지"),
    )

    ui = {}
    if provider == "hf_api":
        st.caption("토큰은 Streamlit Secrets 권장. 아래 입력은 현재 세션에서만 사용됩니다.")
        ui["hf_token"] = st.text_input("HF_TOKEN", value="", type="password", placeholder="hf_... (비우면 Secrets/.env 사용)")
        ui["hf_router_model"] = _model_picker("HF 모델", HF_MODEL_PRESETS, env_cfg.hf_router_model, "hf")
        ui["hf_model_candidates"] = st.text_area("Fallback 후보 (쉼표 구분)", value=env_cfg.hf_model_candidates, height=90)
        ui["hf_max_tokens"] = st.number_input("HF_MAX_TOKENS", 300, 8000, int(env_cfg.hf_max_tokens), 100)
        ui["hf_temperature"] = st.number_input("HF_TEMPERATURE", 0.0, 1.5, float(env_cfg.hf_temperature), 0.1)
        ui["hf_timeout_read"] = st.number_input("HF_TIMEOUT_READ", 30, 600, int(env_cfg.hf_timeout_read), 10)
        ui["hf_max_retries"] = st.number_input("HF_MAX_RETRIES", 1, 10, int(env_cfg.hf_max_retries), 1)
    elif provider == "claude_cli":
        if claude_info["found"]:
            st.success(f"Claude Code CLI 감지: {claude_info['version'] or claude_info['path']}")
        else:
            st.warning("Claude Code CLI 를 찾지 못했습니다. 로컬 노트북에서 `npm i -g @anthropic-ai/claude-code` 설치 후 "
                       "`claude` 로 로그인하세요. (Streamlit Cloud 에서는 사용 불가 → HF Router 사용)")
        ui["claude_model"] = _model_picker("Claude 모델", CLAUDE_MODEL_PRESETS, env_cfg.claude_model, "claude")
        ui["claude_timeout"] = st.number_input("CLI 타임아웃(초)", 30, 900, int(env_cfg.claude_timeout), 10)

    ui["llm_provider"] = provider
    include_llm = provider != "none" and st.checkbox("AI 리뷰 보드 실행 (분석가→비평가→자문)", value=True)


def build_cfg() -> AppConfig:
    """Secrets/.env 기본값 위에 화면 입력값을 현재 실행에만 덮어씁니다."""
    cfg = AppConfig.from_env(
        target_round=int(target), n_sets=int(n_sets), candidate_samples=int(samples),
        backtest_window=int(backtest), max_overlap=int(max_overlap), output_dir="outputs",
        optimize=optimize, tune_ratio=float(tune_ratio), filter_coverage=float(filter_coverage),
        explore_temperature=float(explore_temperature), coverage_weight=float(coverage_weight),
        score_weights=dict(sw),
    )
    for k, v in ui.items():
        if isinstance(v, str) and not v.strip():
            continue  # 빈 입력 → Secrets/.env 값 유지
        setattr(cfg, k, v.strip() if isinstance(v, str) else v)
    return cfg


with st.sidebar:
    if provider != "none" and st.button("🔌 모델 연결 테스트"):
        c = build_cfg()
        client = HfRouterClient(c) if provider == "hf_api" else ClaudeCliClient(c)
        with st.spinner("연결 테스트 중..."):
            st.session_state["llm_test"] = client.test_connection()
    if "llm_test" in st.session_state:
        r = st.session_state["llm_test"]
        if r.get("ok"):
            st.success(f"연결 성공: {r.get('model')}")
            st.caption(str(r.get("content", ""))[:300])
        else:
            st.error(r.get("error", "연결 실패"))

    with st.expander("Streamlit Secrets 예시(TOML)"):
        st.code(Path("STREAMLIT_SECRETS_TEMPLATE.toml").read_text(encoding="utf-8")
                if Path("STREAMLIT_SECRETS_TEMPLATE.toml").exists() else "", language="toml")


def run_harness(cfg: AppConfig) -> HarnessContext:
    csv_path = _save_upload(csv_file, "data.csv") or "data/korealotto_1224V_CSV2.csv"
    xls_path = _save_upload(xls_file, "special.xls") or "data/LottoSpecial_공나온순서_1225.xls"
    status = st.status("에이전트 팀 실행 중...", expanded=True)
    ctx = HarnessContext(cfg=cfg, csv_path=csv_path, xls_path=xls_path, on_progress=lambda m: status.write(m))
    ctx = AgentHarness(include_llm=include_llm).run(ctx)
    status.update(label="완료", state="complete", expanded=False)
    return ctx


col_run, col_apply = st.columns([2, 1])
with col_run:
    if st.button("🚀 Agent Harness 실행", type="primary", width="stretch"):
        st.session_state["ctx"] = run_harness(build_cfg())

prev = st.session_state.get("ctx")
suggestions = (prev.artifacts.get("llm") or {}).get("suggestions") if prev else None
with col_apply:
    if st.button("🤖 AI 자문 제안 적용 후 재실행", disabled=not suggestions, width="stretch"):
        st.session_state["ctx"] = run_harness(apply_suggestions(build_cfg(), suggestions))
        st.session_state["applied"] = suggestions

ctx = st.session_state.get("ctx")
tabs = st.tabs(["최종 결과", "최적화·감사", "AI 리뷰", "백테스트", "데이터/후보", "하네스 설계/다운로드"])

if not ctx:
    with tabs[0]:
        st.info("왼쪽에서 설정 후 'Agent Harness 실행'을 누르세요.")
    with tabs[5]:
        st.dataframe(pd.DataFrame([s.__dict__ for s in agent_team_spec()]), width="stretch")
    st.stop()

result = ctx.artifacts.get("pipeline", {})
opt = result.get("optimization", {})
audit = ctx.artifacts.get("audit", {})
llm = ctx.artifacts.get("llm") or {}

with tabs[0]:
    st.subheader(f"{ctx.cfg.target_round}회차 후보 {ctx.cfg.n_sets}세트")
    if st.session_state.get("applied"):
        st.caption(f"AI 자문 제안 적용됨: `{json.dumps(st.session_state['applied'], ensure_ascii=False)}`")
    sets = result.get("final_sets", [])
    cols = st.columns(max(1, len(sets)))
    for i, nums in enumerate(sets):
        with cols[i]:
            st.metric(f"Set {i+1}", " · ".join(map(str, nums)))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("holdout edge (nats)", f"{opt.get('holdout_edge_nats', 0):+.4f}")
    m2.metric("holdout 평균 적중", f"{opt.get('holdout_mean_hits', 0):.3f}", f"{opt.get('holdout_mean_hits', 0) - 0.8:+.3f} vs 무작위")
    m3.metric("앙상블 적중 p-value", f"{opt.get('ensemble_hits_p_value', 1):.3f}")
    m4.metric("감사 판정", audit.get("verdict", "-"))
    st.warning("공정추첨에서 모든 6개 조합의 1등 확률은 1/8,145,060 로 동일합니다. "
               "본 결과는 통계 연구용 후보이며, 감사 판정이 무작위 대비 우위를 보여주지 못하면 신호는 자동 수축됩니다.")
    st.subheader("에이전트 실행 로그")
    st.dataframe(pd.DataFrame([a.__dict__ for a in ctx.agent_results]), width="stretch")

with tabs[1]:
    st.subheader("통계 감사 (결정론적)")
    for f in audit.get("risk_flags", []):
        st.write("⚠️", f)
    st.json({k: v for k, v in audit.items() if k != "risk_flags"})
    st.subheader("앙상블 가중치")
    w = result.get("weights")
    if isinstance(w, pd.DataFrame):
        st.bar_chart(w.set_index("strategy")["weight"])
    st.json({k: v for k, v in opt.items() if k != "weights"})
    st.subheader("학습형 구조 필터")
    st.dataframe(pd.DataFrame(result.get("filters", {})).T, width="stretch")

with tabs[2]:
    if not llm.get("ok"):
        st.info(llm.get("error") or "AI 리뷰를 실행하려면 공급자를 선택하고 'AI 리뷰 보드 실행'을 켜세요.")
    else:
        st.caption(f"공급자: {PROVIDERS.get(llm['provider'], llm['provider'])} · 모델: `{llm['model']}`")
        steps = llm.get("steps", {})
        st.subheader("1) 분석가")
        st.markdown(steps.get("analyst", {}).get("content") or "-")
        st.subheader("2) 비평가")
        st.json(llm.get("critic") or {"raw": steps.get("critic", {}).get("content")})
        st.subheader("3) 최적화 자문")
        st.json(llm.get("advisor") or {"raw": steps.get("advisor", {}).get("content")})
        st.write("적용 가능한 제안(허용 범위 클램프):", llm.get("suggestions") or "없음")
        costs = [s.get("cost_usd") for s in steps.values() if s.get("cost_usd")]
        if costs:
            st.caption(f"Claude CLI 비용 합계: ${sum(costs):.4f}")

with tabs[3]:
    st.subheader("전략별 walk-forward 요약 (무작위 기대 적중 0.8)")
    st.dataframe(result.get("backtest_summary"), width="stretch")
    detail = result.get("backtest_detail")
    if isinstance(detail, pd.DataFrame) and not detail.empty:
        roll = detail.pivot_table(index="round", columns="strategy", values="hits").rolling(30, min_periods=5).mean()
        st.caption("전략별 30회 이동평균 적중 수")
        st.line_chart(roll)

with tabs[4]:
    st.subheader("번호별 점수/확률")
    st.dataframe(result.get("number_scores"), width="stretch")
    st.subheader("상위 후보 조합")
    cands = result.get("candidates")
    st.dataframe(cands.head(200) if cands is not None else None, width="stretch")

with tabs[5]:
    st.dataframe(pd.DataFrame([s.__dict__ for s in agent_team_spec()]), width="stretch")
    st.subheader("다운로드")
    for name, path in ctx.artifacts.get("paths", {}).items():
        p = Path(path)
        if p.exists():
            st.download_button(name, p.read_bytes(), file_name=p.name, key=f"dl_{name}")
