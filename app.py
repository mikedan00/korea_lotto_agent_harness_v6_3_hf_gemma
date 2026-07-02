from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness, agent_team_spec
from src.llm_engine import HfRouterClient

st.set_page_config(page_title="Korea Lotto Agent Harness v6.3 HF Gemma", layout="wide")
st.title("🎯 Korea Lotto 6/45 Agent Harness v6.3 · HF Gemma4")
st.caption("CLI 없이 Streamlit 화면에서 HF_TOKEN과 Gemma4 Router 모델을 사용해 통계 분석 + LLM 보조 분석을 실행합니다.")

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
DEFAULT_CANDIDATES = (
    "google/gemma-4-26B-A4B-it:deepinfra,"
    "google/gemma-4-26B-A4B-it:novita,"
    "google/gemma-4-31B-it:deepinfra,"
    "google/gemma-4-31B-it:together,"
    "Qwen/Qwen3.5-9B:together,"
    "Qwen/Qwen2.5-7B-Instruct:together"
)


def _save_upload(upload, suffix: str) -> str | None:
    if upload is None:
        return None
    p = Path(tempfile.gettempdir()) / f"lotto_upload_{suffix}"
    p.write_bytes(upload.getvalue())
    return str(p)


def build_cfg_from_ui(
    target: int,
    n_sets: int,
    samples: int,
    backtest: int,
    max_overlap: int,
    ui_token: str,
    ui_model: str,
    ui_candidates: str,
    ui_max_tokens: int,
    ui_temperature: float,
    ui_connect_timeout: int,
    ui_read_timeout: int,
    ui_retries: int,
) -> AppConfig:
    """Streamlit Secrets/.env를 먼저 읽고, 화면 입력값이 있으면 그 값을 현재 실행에만 덮어씁니다."""
    cfg = AppConfig.from_env(
        target_round=int(target),
        n_sets=int(n_sets),
        candidate_samples=int(samples),
        backtest_window=int(backtest),
        max_overlap=int(max_overlap),
        output_dir="outputs",
    )
    cfg.llm_engine = "hf_api"
    if ui_token.strip():
        cfg.hf_token = ui_token.strip()
    if ui_model.strip():
        cfg.hf_router_model = ui_model.strip()
    if ui_candidates.strip():
        cfg.hf_model_candidates = ui_candidates.strip()
    cfg.hf_max_tokens = int(ui_max_tokens)
    cfg.hf_temperature = float(ui_temperature)
    cfg.hf_timeout_connect = int(ui_connect_timeout)
    cfg.hf_timeout_read = int(ui_read_timeout)
    cfg.hf_max_retries = int(ui_retries)
    return cfg


with st.sidebar:
    st.header("실행 설정")
    target = st.number_input("목표 회차", min_value=1, value=1226, step=1)
    n_sets = st.number_input("최종 세트 수", min_value=1, max_value=10, value=2, step=1)
    samples = st.number_input("후보 조합 샘플 수", min_value=1000, max_value=200000, value=12000, step=1000)
    backtest = st.number_input("최근 백테스트 회차", min_value=30, max_value=1000, value=180, step=10)
    max_overlap = st.number_input("세트 간 최대 중복", min_value=0, max_value=5, value=2, step=1)

    st.divider()
    st.subheader("데이터")
    csv_file = st.file_uploader("기본 CSV 업로드", type=["csv"])
    xls_file = st.file_uploader("1225 공나온순서 XLS/HTML 업로드", type=["xls", "html"])

    st.divider()
    st.subheader("HF Gemma4 설정")
    env_cfg = AppConfig.from_env()
    st.caption("토큰은 Streamlit Secrets에 넣는 것을 권장합니다. 아래 입력란은 현재 실행 세션에서만 임시 사용됩니다.")
    ui_token = st.text_input("HF_TOKEN", value="", type="password", placeholder="hf_...", help="비워두면 Streamlit Secrets/.env의 HF_TOKEN을 사용합니다.")
    ui_model = st.text_input("HF_ROUTER_MODEL", value=env_cfg.hf_router_model or DEFAULT_MODEL)
    ui_candidates = st.text_area("HF_MODEL_CANDIDATES", value=env_cfg.hf_model_candidates or DEFAULT_CANDIDATES, height=120)
    ui_max_tokens = st.number_input("HF_MAX_TOKENS", min_value=300, max_value=4000, value=int(env_cfg.hf_max_tokens or 1400), step=100)
    ui_temperature = st.number_input("HF_TEMPERATURE", min_value=0.0, max_value=1.5, value=float(env_cfg.hf_temperature or 0.2), step=0.1)
    ui_connect_timeout = st.number_input("HF_TIMEOUT_CONNECT", min_value=3, max_value=60, value=int(env_cfg.hf_timeout_connect or 10), step=1)
    ui_read_timeout = st.number_input("HF_TIMEOUT_READ", min_value=30, max_value=300, value=int(env_cfg.hf_timeout_read or 120), step=10)
    ui_retries = st.number_input("HF_MAX_RETRIES", min_value=1, max_value=10, value=int(env_cfg.hf_max_retries or 3), step=1)

    tmp_cfg = build_cfg_from_ui(target, n_sets, samples, backtest, max_overlap, ui_token, ui_model, ui_candidates, ui_max_tokens, ui_temperature, ui_connect_timeout, ui_read_timeout, ui_retries)
    st.write("LLM_ENGINE:", tmp_cfg.llm_engine)
    st.write("HF_TOKEN:", "설정됨 ✅" if tmp_cfg.hf_token else "미설정 ⚠️")
    st.write("1순위 모델:", tmp_cfg.model_candidates[0] if tmp_cfg.model_candidates else "없음")

    col_a, col_b = st.columns(2)
    with col_a:
        include_llm = st.checkbox("Gemma4 LLM 분석 포함", value=bool(tmp_cfg.hf_token))
    with col_b:
        if st.button("HF 연결 테스트"):
            with st.spinner("HF Router 연결 테스트 중..."):
                r = HfRouterClient(tmp_cfg).test_connection()
            st.session_state["hf_test"] = r
    if "hf_test" in st.session_state:
        r = st.session_state["hf_test"]
        if r.get("ok"):
            st.success(f"연결 성공: {r.get('model')}")
            st.caption(str(r.get("content", ""))[:300])
        else:
            st.error(r.get("error", "연결 실패"))

    with st.expander("Streamlit Secrets에 넣을 TOML"):
        st.code(
            'LLM_ENGINE = "hf_api"\n'
            'HF_TOKEN = "hf_새로_발급한_토큰"\n'
            f'HF_ROUTER_MODEL = "{DEFAULT_MODEL}"\n'
            f'HF_MODEL_CANDIDATES = "{DEFAULT_CANDIDATES}"\n'
            'HF_MAX_TOKENS = "1400"\n'
            'HF_TEMPERATURE = "0.2"\n'
            'HF_TIMEOUT_CONNECT = "10"\n'
            'HF_TIMEOUT_READ = "120"\n'
            'HF_MAX_RETRIES = "3"\n',
            language="toml",
        )

run = st.button("🚀 Streamlit에서 Agent Harness + Gemma4 실행", type="primary")

if run:
    csv_path = _save_upload(csv_file, "data.csv") or "data/korealotto_1224V_CSV2.csv"
    xls_path = _save_upload(xls_file, "special.xls") or "data/LottoSpecial_공나온순서_1225.xls"
    cfg = build_cfg_from_ui(target, n_sets, samples, backtest, max_overlap, ui_token, ui_model, ui_candidates, ui_max_tokens, ui_temperature, ui_connect_timeout, ui_read_timeout, ui_retries)
    ctx = HarnessContext(cfg=cfg, csv_path=csv_path, xls_path=xls_path)
    with st.spinner("에이전트 팀 실행 중... 통계 백테스트 → 후보 생성 → 검증 → Gemma4 분석"):
        ctx = AgentHarness(include_llm=include_llm).run(ctx)
    st.session_state["ctx"] = ctx

ctx = st.session_state.get("ctx")

spec_tab, result_tab, backtest_tab, data_tab, report_tab = st.tabs(["하네스 설계", "최종 결과", "백테스트", "데이터/후보", "Gemma4/다운로드"])

with spec_tab:
    st.subheader("에이전트 팀 설계")
    spec_df = pd.DataFrame([s.__dict__ for s in agent_team_spec()])
    st.dataframe(spec_df, use_container_width=True)
    st.markdown(
        """
        **v6.3 변경점**
        - CLI 실행 없이 Streamlit 화면에서 HF_TOKEN과 Gemma4 Router 모델을 직접 입력해 실행
        - Streamlit Secrets → 화면 입력값 → 기본값 순으로 LLM 설정 적용
        - HF 연결 테스트 버튼 추가
        - LLM은 번호를 단독 생성하지 않고, 통계 엔진 결과를 비판·설명·리스크 요약하는 Gemma4 분석 에이전트로 사용
        """
    )

if ctx:
    result = ctx.artifacts.get("pipeline", {})
    with result_tab:
        st.subheader(f"{ctx.cfg.target_round}회차 최종 후보 {ctx.cfg.n_sets}세트")
        cols = st.columns(max(1, len(result.get("final_sets", []))))
        for i, nums in enumerate(result.get("final_sets", [])):
            with cols[i]:
                st.metric(f"Set {i+1}", " · ".join(map(str, nums)))
        st.write("정확확률 요약", result.get("probability", {}))
        st.warning("공정추첨을 전제로 모든 6개 조합의 실제 1등 확률은 동일합니다. 본 결과는 통계 연구용 후보입니다.")
        st.subheader("에이전트 실행 로그")
        st.dataframe(pd.DataFrame([a.__dict__ for a in ctx.agent_results]), use_container_width=True)

    with backtest_tab:
        st.subheader("전략별 rolling-forward 요약")
        st.dataframe(result.get("backtest_summary"), use_container_width=True)
        st.subheader("앙상블 가중치")
        st.dataframe(result.get("weights"), use_container_width=True)

    with data_tab:
        st.subheader("번호별 앙상블 점수")
        st.dataframe(result.get("number_scores"), use_container_width=True)
        st.subheader("상위 후보 조합")
        candidates = result.get("candidates")
        st.dataframe(candidates.head(100) if candidates is not None else None, use_container_width=True)

    with report_tab:
        if "llm" in ctx.artifacts:
            st.subheader("Gemma4 보조 분석")
            llm = ctx.artifacts["llm"]
            st.json(llm if not llm.get("content") else {"ok": llm.get("ok"), "model": llm.get("model")})
            if llm.get("content"):
                st.markdown(llm["content"])
        else:
            st.info("Gemma4 LLM 분석을 포함하려면 왼쪽에서 HF_TOKEN을 설정하고 'Gemma4 LLM 분석 포함'을 켜세요.")

        st.subheader("다운로드")
        paths = ctx.artifacts.get("paths", {})
        for name, path in paths.items():
            p = Path(path)
            if p.exists():
                st.download_button(name, p.read_bytes(), file_name=p.name)
else:
    with result_tab:
        st.info("왼쪽 설정 후 'Streamlit에서 Agent Harness + Gemma4 실행'을 누르세요.")
