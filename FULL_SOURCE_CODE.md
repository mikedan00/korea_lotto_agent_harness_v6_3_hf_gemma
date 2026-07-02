# Korea Lotto Agent Harness v6.3 HF Gemma4 - Full Source


## .streamlit/config.toml

```toml
[theme]
base="light"
primaryColor="#2563eb"

[server]
maxUploadSize=50

```


## FULL_SOURCE_CODE.md

```md
# Korea Lotto Agent Harness v6.2 Prime Factor - Full Source Code


# FILE: .streamlit/config.toml

```toml
[theme]
base="light"
primaryColor="#2563eb"

[server]
maxUploadSize=50

```


# FILE: README.md

```md
# Korea Lotto Agent Harness v6

한국 로또 6/45 연구용 분석 하네스입니다. 기획·데이터·전략·백테스트·검증·LLM 분석·리포트 에이전트 팀을 구성해 1226회차 후보 2세트를 생성합니다.

> 공정추첨을 전제로 모든 6개 조합의 실제 1등 확률은 동일합니다. 이 프로그램은 당첨 보장이 아니라 통계 연구와 백테스트 비교를 위한 도구입니다.

## VS Code 로컬 실행

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python cli.py --csv data/korealotto_1224V_CSV2.csv --xls data/LottoSpecial_공나온순서_1225.xls --target 1226 --sets 2 --samples 12000 --outdir outputs
streamlit run app.py
```

## Streamlit Cloud Secrets

`Manage app > Settings > Secrets`에 아래 TOML을 넣으세요.

```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_새로_발급한_토큰"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = "1400"
HF_TEMPERATURE = "0.2"
HF_TIMEOUT_CONNECT = "10"
HF_TIMEOUT_READ = "120"
HF_MAX_RETRIES = "3"
```

## CLI 출력

```text
Set 1: ...
Set 2: ...
```

## 하네스 구성

자세한 내용은 `docs/AGENT_TEAM_DESIGN.md`, `docs/HARNESS_SPEC.md`, `UPDATE_STRATEGY.md`를 참고하세요.


## Colab import 오류 해결

`ModuleNotFoundError: No module named 'src'` 오류가 나면 `notebooks/lotto_agent_harness_colab.ipynb`의 첫 셀을 다시 실행하세요. v6.1 노트북은 ZIP 업로드, 압축 해제, 프로젝트 루트 탐색, `sys.path` 등록을 자동으로 처리합니다.


## v6.2 소인수분해 전략 추가

이번 버전은 `PrimeFactorStrategy`를 추가했습니다. 당첨번호 6개를 소인수분해하여 소인수별 장기/최근/gap 신호를 계산하고, 후보 조합 단계에서 prime count, omega sum, square-free count, semiprime count, 2/3/5/7 배수 구조가 과거 분포와 얼마나 가까운지 보조점수로 반영합니다. 이 전략은 연구용 feature이며, 당첨 보장을 의미하지 않습니다.

```


# FILE: STREAMLIT_SECRETS_TEMPLATE.toml

```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_새로_발급한_토큰"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = "1400"
HF_TEMPERATURE = "0.2"
HF_TIMEOUT_CONNECT = "10"
HF_TIMEOUT_READ = "120"
HF_MAX_RETRIES = "3"

```


# FILE: UPDATE_STRATEGY.md

```md
# 단계별 업데이트 전략

## v6.0 Agent Harness
- 에이전트 팀 구조 도입
- 데이터/전략/백테스트/검증/리포트 분리
- Streamlit + CLI + Colab 공통 엔진 구성

## v6.1 공식 데이터 자동 동기화
- 동행복권 공식 엑셀/페이지 동기화 모듈 추가
- 원본 해시와 데이터 manifest 저장

## v6.2 확률 보정 강화
- isotonic/sigmoid calibration 실험
- Brier/AP/Calibration curve 시각화

## v6.3 백테스트 하네스 확장
- full expanding-window
- permutation test
- bootstrap confidence interval

## v6.4 AutoML/Optuna 연구 모드
- strategy weight 탐색
- 후보 조합 생성 하이퍼파라미터 탐색

## v6.5 서버 프록시
- HF_TOKEN 보호용 FastAPI proxy
- 사용자별 quota/rate-limit

## v6.6 Android 앱 연동
- Android 앱이 local prediction + proxy LLM 분석 호출


## v6.2 소인수분해 전략 추가

이번 버전은 `PrimeFactorStrategy`를 추가했습니다. 당첨번호 6개를 소인수분해하여 소인수별 장기/최근/gap 신호를 계산하고, 후보 조합 단계에서 prime count, omega sum, square-free count, semiprime count, 2/3/5/7 배수 구조가 과거 분포와 얼마나 가까운지 보조점수로 반영합니다. 이 전략은 연구용 feature이며, 당첨 보장을 의미하지 않습니다.

```


# FILE: app.py

```py
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness, agent_team_spec

st.set_page_config(page_title="Korea Lotto Agent Harness v6", layout="wide")
st.title("🎯 Korea Lotto 6/45 Agent Harness v6")
st.caption("기획·구현·검증·백테스팅·리포트 에이전트 팀으로 실행되는 연구용 로또 분석 하네스")

with st.sidebar:
    st.header("실행 설정")
    target = st.number_input("목표 회차", min_value=1, value=1226, step=1)
    n_sets = st.number_input("최종 세트 수", min_value=1, max_value=10, value=2, step=1)
    samples = st.number_input("후보 조합 샘플 수", min_value=1000, max_value=200000, value=12000, step=1000)
    backtest = st.number_input("최근 백테스트 회차", min_value=30, max_value=1000, value=180, step=10)
    max_overlap = st.number_input("세트 간 최대 중복", min_value=0, max_value=5, value=2, step=1)
    include_llm = st.checkbox("HF LLM 보조 분석 실행", value=False)

    st.divider()
    st.subheader("데이터")
    csv_file = st.file_uploader("기본 CSV 업로드", type=["csv"])
    xls_file = st.file_uploader("1225 공나온순서 XLS/HTML 업로드", type=["xls", "html"])

    st.divider()
    st.subheader("LLM 설정 상태")
    tmp_cfg = AppConfig.from_env()
    st.write("LLM_ENGINE:", tmp_cfg.llm_engine)
    st.write("HF_TOKEN:", "설정됨 ✅" if tmp_cfg.hf_token else "미설정 ⚠️")
    st.code(Path("STREAMLIT_SECRETS_TEMPLATE.toml").read_text(encoding="utf-8") if Path("STREAMLIT_SECRETS_TEMPLATE.toml").exists() else "", language="toml")


def _save_upload(upload, suffix: str) -> str | None:
    if upload is None:
        return None
    p = Path(tempfile.gettempdir()) / f"lotto_upload_{suffix}"
    p.write_bytes(upload.getvalue())
    return str(p)

run = st.button("🚀 Agent Harness 실행", type="primary")

if run:
    csv_path = _save_upload(csv_file, "data.csv") or "data/korealotto_1224V_CSV2.csv"
    xls_path = _save_upload(xls_file, "special.xls") or "data/LottoSpecial_공나온순서_1225.xls"
    cfg = AppConfig.from_env(target_round=int(target), n_sets=int(n_sets), candidate_samples=int(samples), backtest_window=int(backtest), max_overlap=int(max_overlap), output_dir="outputs")
    ctx = HarnessContext(cfg=cfg, csv_path=csv_path, xls_path=xls_path)
    with st.spinner("에이전트 팀 실행 중..."):
        ctx = AgentHarness(include_llm=include_llm).run(ctx)
    st.session_state["ctx"] = ctx

ctx = st.session_state.get("ctx")

spec_tab, result_tab, backtest_tab, data_tab, report_tab = st.tabs(["하네스 설계", "최종 결과", "백테스트", "데이터/후보", "리포트/다운로드"])

with spec_tab:
    st.subheader("에이전트 팀 설계")
    spec_df = pd.DataFrame([s.__dict__ for s in agent_team_spec()])
    st.dataframe(spec_df, use_container_width=True)
    st.markdown("""
    **하네스 품질 게이트**
    - 데이터 유효성 오류 0건
    - rolling-forward 방식으로 미래 데이터 누수 방지
    - 전략별 성과를 softmax 앙상블 가중치로 변환
    - 최종 세트 간 중복 번호 제한
    - LLM은 예측 엔진이 아니라 설명·비판·리스크 요약 보조기로 사용
    """)

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
        st.dataframe(result.get("candidates").head(100), use_container_width=True)

    with report_tab:
        paths = ctx.artifacts.get("paths", {})
        st.subheader("다운로드")
        for name, path in paths.items():
            p = Path(path)
            if p.exists():
                st.download_button(name, p.read_bytes(), file_name=p.name)
        if "llm" in ctx.artifacts:
            st.subheader("LLM 보조 분석")
            st.json(ctx.artifacts["llm"] if not ctx.artifacts["llm"].get("content") else {"ok": ctx.artifacts["llm"].get("ok"), "model": ctx.artifacts["llm"].get("model")})
            if ctx.artifacts["llm"].get("content"):
                st.markdown(ctx.artifacts["llm"]["content"])
else:
    with result_tab:
        st.info("왼쪽 설정 후 'Agent Harness 실행'을 누르세요.")

```


# FILE: cli.py

```py
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness


def main():
    ap = argparse.ArgumentParser(description="Korea Lotto 6/45 Agent Harness v6")
    ap.add_argument("--csv", default="data/korealotto_1224V_CSV2.csv")
    ap.add_argument("--xls", default="data/LottoSpecial_공나온순서_1225.xls")
    ap.add_argument("--target", type=int, default=1226)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--backtest", type=int, default=180)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig.from_env(
        target_round=args.target,
        n_sets=args.sets,
        candidate_samples=args.samples,
        backtest_window=args.backtest,
        output_dir=args.outdir,
    )
    ctx = HarnessContext(cfg=cfg, csv_path=args.csv, xls_path=args.xls)
    ctx = AgentHarness(include_llm=args.llm).run(ctx)

    result = ctx.artifacts.get("pipeline", {})
    print(f"\n[1226 후보 {cfg.n_sets}세트]")
    for i, s in enumerate(result.get("final_sets", []), 1):
        print(f"Set {i}: {', '.join(map(str, s))}")
    print("\n[Agent Results]")
    for ar in ctx.agent_results:
        print(f"- {ar.name}: {ar.status} | {ar.message}")
    print(f"\noutputs: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()

```


# FILE: docs/AGENT_TEAM_DESIGN.md

```md
# 에이전트 팀 설계

이 프로젝트는 단일 함수형 예측기가 아니라 **Agent Harness**입니다. 각 에이전트는 입력, 출력, 품질 게이트를 갖고 순차적으로 실행됩니다.

## 팀 구성

| 에이전트 | 역할 | 핵심 산출물 |
|---|---|---|
| PlannerAgent | 연구 계획, 품질 게이트 정의 | plan |
| DataIngestionAgent | CSV/XLS 로딩, 정규화, 검증 | draws |
| FeatureBacktestAgent | 전략 구현, rolling-forward 백테스트, 후보 생성 | pipeline |
| VerificationAgent | 최종 세트 유효성, 중복도, 확률 요약 검증 | verification |
| LlmAnalystAgent | HF Router 기반 보조 해석 | llm |
| ReportAgent | CSV/JSON 산출물 저장 | outputs |

## 하네스 철학

- 번호 추천보다 **검증 가능한 실험 루틴**이 우선입니다.
- rolling-forward 외의 무작위 K-fold는 기본 금지합니다.
- LLM은 예측 엔진이 아니라 리포트/비판/설명 계층입니다.
- 당첨 보장 표현은 모든 산출물에서 금지합니다.

```


# FILE: docs/HARNESS_SPEC.md

```md
# Harness Specification

## 입력

- `korealotto_1224V_CSV2.csv`: 1~1224회 기본 회차 데이터
- `LottoSpecial_공나온순서_1225.xls`: 1225회 포함 공나온순서 HTML/XLS 파일
- Streamlit 업로드 파일 또는 CLI 경로

## 출력

- `prediction_result.json`
- `harness_report.json`
- `backtest_summary.csv`
- `backtest_detail.csv`
- `weights.csv`
- `number_scores.csv`
- `candidates.csv`

## 품질 게이트

1. 번호 범위 1~45
2. 본번호 6개 중복 없음
3. 보너스와 본번호 중복 없음
4. 회차 병합 후 정렬
5. 백테스트는 t 시점에서 t+1만 예측
6. 최종 세트 수 일치
7. 세트 간 overlap 제한 준수

```


# FILE: docs/PRIME_FACTOR_STRATEGY.md

```md
# Prime Factor Strategy v6.2

이 문서는 로또 번호를 소인수분해하여 연구용 예측 feature로 사용하는 전략을 설명합니다.

## 핵심 아이디어

각 번호 `1~45`를 소인수 지수 벡터로 변환합니다. 예를 들면 `12 = 2^2 * 3`, `27 = 3^3`, `38 = 2 * 19`, `45 = 3^2 * 5`입니다. 그 후 회차별 당첨번호 6개를 다음 관점에서 분석합니다.

- 소인수별 장기 출현량
- 소인수별 최근 60회 출현량
- 소인수별 현재 gap / 평균 gap
- prime number count
- composite count
- omega sum: 소인수 지수 총합
- square-free count
- semiprime count
- 2, 3, 5, 7로 나누어지는 숫자 개수

## 구현 위치

- `src/strategies.py`
  - `PrimeFactorStrategy`
  - `prime_factor_exponents`
  - `draw_prime_profile`
  - `prime_combo_structural_score`

- `src/engine.py`
  - 후보 조합 scoring에 `prime_struct_score` 반영
  - 후보 테이블에 `prime_count`, `omega_sum`, `squarefree_count`, `semiprime_count` 출력

## 주의

이 전략은 과거 데이터의 소인수 구조를 feature로 추가하는 연구용 알고리즘입니다. 공정 추첨에서 특정 조합의 물리적 당첨확률을 바꾸지는 않습니다. 신뢰도는 반드시 rolling-forward 백테스트 결과로만 판단해야 합니다.

```


# FILE: notebooks/colab_single_file.py

```py
# Google Colab / Jupyter 실행용: 프로젝트 루트 자동 탐색 + src import 오류 방지
!pip -q install pandas numpy matplotlib requests python-dotenv lxml openpyxl streamlit

from pathlib import Path
import zipfile, os, sys, json, shutil

PROJECT_NAME = "korea_lotto_agent_harness_v6"
ZIP_NAMES = [
    "/content/korea_lotto_agent_harness_v6.zip",
    "/content/korea_lotto_agent_harness_v6_1_colab_fix.zip",
    str(Path.cwd() / "korea_lotto_agent_harness_v6.zip"),
    str(Path.cwd() / "korea_lotto_agent_harness_v6_1_colab_fix.zip"),
]

# 1) ZIP이 있으면 압축 해제
zip_found = None
for zp in ZIP_NAMES:
    if Path(zp).exists():
        zip_found = Path(zp)
        break

# Colab에서 ZIP을 아직 업로드하지 않았다면 업로드 창 표시
if zip_found is None and Path('/content').exists():
    try:
        from google.colab import files
        print("ZIP 파일을 업로드하세요: korea_lotto_agent_harness_v6.zip")
        uploaded = files.upload()
        for name in uploaded.keys():
            if name.endswith('.zip'):
                zip_found = Path('/content') / name
                break
    except Exception as e:
        print("Colab files.upload()를 사용할 수 없습니다:", e)

if zip_found is not None:
    print("압축 해제:", zip_found)
    with zipfile.ZipFile(zip_found, "r") as z:
        z.extractall("/content" if Path('/content').exists() else Path.cwd())
else:
    print("ZIP 파일을 찾지 못했습니다. 이미 압축 해제된 폴더를 탐색합니다.")

# 2) src/config.py가 있는 실제 프로젝트 루트 자동 탐색
def find_project_root():
    roots = []
    for base in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if base.exists():
            roots.append(base)
    seen = set()
    for base in roots:
        for p in base.rglob('src/config.py'):
            root = p.parent.parent
            if root not in seen:
                seen.add(root)
                # app.py 또는 cli.py가 있는 프로젝트를 우선 선택
                if (root / 'app.py').exists() or (root / 'cli.py').exists():
                    return root
    return None

PROJECT = find_project_root()
if PROJECT is None:
    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. ZIP을 /content에 업로드했는지, "
        "압축 해제 후 src/config.py가 존재하는지 확인하세요."
    )

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("프로젝트 루트:", PROJECT)
print("현재 작업 폴더:", Path.cwd())
print("src/config.py 존재:", (PROJECT / 'src' / 'config.py').exists())

# 3) 기본 실행
from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness

cfg = AppConfig.from_env(
    target_round=1226,
    n_sets=2,
    candidate_samples=12000,
    backtest_window=180,
    output_dir="/content/lotto_outputs" if Path('/content').exists() else str(PROJECT / 'outputs_colab')
)
ctx = HarnessContext(
    cfg=cfg,
    csv_path=str(PROJECT / "data" / "korealotto_1224V_CSV2.csv"),
    xls_path=str(PROJECT / "data" / "LottoSpecial_공나온순서_1225.xls")
)
ctx = AgentHarness(include_llm=False).run(ctx)

result = ctx.artifacts["pipeline"]
print("\n1226회차 후보 2세트")
for i, s in enumerate(result["final_sets"], 1):
    print(f"Set {i}: {s}")

print("\n에이전트 실행 결과")
for ar in ctx.agent_results:
    print(ar.name, ar.status, ar.message)

print("\n저장 경로:", cfg.output_dir)
print([str(p) for p in Path(cfg.output_dir).glob("*")])
```


# FILE: pyproject.toml

```toml
[project]
name = "korea-lotto-agent-harness-v6"
version = "6.0.0"
description = "Korean Lotto 6/45 research harness with agent-team orchestration, rolling backtest, Streamlit, Colab, and HF LLM hooks."
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

```


# FILE: requirements.txt

```txt
streamlit>=1.32
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
requests>=2.31
python-dotenv>=1.0
lxml>=4.9
openpyxl>=3.1

```


# FILE: src/__init__.py

```py

```


# FILE: src/config.py

```py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _streamlit_secret(key: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        return None
    return None


def get_setting(key: str, default: str = "") -> str:
    val = _streamlit_secret(key)
    if val is not None and val != "":
        return val
    return os.getenv(key, default)


@dataclass
class AppConfig:
    target_round: int = 1226
    n_sets: int = 2
    candidate_samples: int = 12000
    backtest_window: int = 180
    min_train_rounds: int = 200
    seed: int = 20260530
    max_overlap: int = 2
    sum_min: int = 98
    sum_max: int = 177
    output_dir: str = "outputs"

    llm_engine: str = "hf_api"
    hf_token: str = ""
    hf_router_model: str = "google/gemma-4-26B-A4B-it:deepinfra"
    hf_model_candidates: str = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
    hf_max_tokens: int = 1400
    hf_temperature: float = 0.2
    hf_timeout_connect: int = 10
    hf_timeout_read: int = 120
    hf_max_retries: int = 3

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppConfig":
        cfg = cls(
            llm_engine=get_setting("LLM_ENGINE", "hf_api"),
            hf_token=get_setting("HF_TOKEN", ""),
            hf_router_model=get_setting("HF_ROUTER_MODEL", get_setting("HF_MODEL_ID", "google/gemma-4-26B-A4B-it:deepinfra")),
            hf_model_candidates=get_setting("HF_MODEL_CANDIDATES", cls.hf_model_candidates),
            hf_max_tokens=int(float(get_setting("HF_MAX_TOKENS", "1400"))),
            hf_temperature=float(get_setting("HF_TEMPERATURE", "0.2")),
            hf_timeout_connect=int(float(get_setting("HF_TIMEOUT_CONNECT", "10"))),
            hf_timeout_read=int(float(get_setting("HF_TIMEOUT_READ", "120"))),
            hf_max_retries=int(float(get_setting("HF_MAX_RETRIES", "3"))),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    @property
    def model_candidates(self) -> list[str]:
        seen = []
        for m in [self.hf_router_model] + [x.strip() for x in self.hf_model_candidates.split(",") if x.strip()]:
            if m not in seen:
                seen.append(m)
        return seen

```


# FILE: src/data_loader.py

```py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class Draw:
    round: int
    date: str
    numbers: tuple[int, int, int, int, int, int]
    bonus: int
    source: str = ""

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "date": self.date,
            "n1": self.numbers[0],
            "n2": self.numbers[1],
            "n3": self.numbers[2],
            "n4": self.numbers[3],
            "n5": self.numbers[4],
            "n6": self.numbers[5],
            "bonus": self.bonus,
            "source": self.source,
        }


def _clean_col(c) -> str:
    return str(c).strip().replace(" ", "")


def _extract_int(value) -> int:
    s = str(value).replace(",", "")
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError(f"정수를 추출할 수 없습니다: {value!r}")
    return int(m.group())


def _read_csv_any(path: str | Path) -> pd.DataFrame:
    last = None
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 로딩 실패: {path}, 마지막 오류={last}")


def load_lotto_csv(path: str | Path) -> list[Draw]:
    df = _read_csv_any(path)
    df.columns = [_clean_col(c) for c in df.columns]
    draw_col = next((c for c in df.columns if c in ["회차", "round", "Round", "draw"]), df.columns[0])
    date_col = next((c for c in df.columns if c in ["추첨일", "당첨일자", "date", "Date"]), None)

    # Korean official-like column names first, then positional fallback.
    ncols = []
    for i in range(1, 7):
        for candidate in [f"당첨번호{i}", f"당번{i}", f"n{i}", f"N{i}"]:
            if candidate in df.columns:
                ncols.append(candidate)
                break
    if len(ncols) < 6:
        numeric_cols = [c for c in df.columns if c not in [draw_col, date_col] and "보" not in c]
        ncols = numeric_cols[:6]
    bonus_col = next((c for c in df.columns if c in ["보너스", "보볼", "bonus", "Bonus"]), None)
    if bonus_col is None:
        candidates = [c for c in df.columns if "보" in c]
        bonus_col = candidates[0] if candidates else df.columns[8]

    draws: list[Draw] = []
    for _, row in df.iterrows():
        try:
            rnd = _extract_int(row[draw_col])
            date = str(row[date_col]) if date_col else ""
            nums = tuple(sorted(_extract_int(row[c]) for c in ncols[:6]))
            bonus = _extract_int(row[bonus_col])
            draws.append(Draw(rnd, date, nums, bonus, source=str(path)))
        except Exception:
            continue
    return validate_draws(draws)


def load_lotto_html_xls(path: str | Path) -> list[Draw]:
    # The uploaded LottoSpecial file is an UTF-8 HTML table with .xls extension.
    tables = pd.read_html(path, encoding="utf-8")
    if not tables:
        return []
    df = tables[0]
    df.columns = [_clean_col(c) for c in df.columns]
    draw_col = next((c for c in df.columns if c in ["회차", "round", "Round"]), df.columns[0])
    date_col = next((c for c in df.columns if c in ["당첨일자", "추첨일", "date", "Date"]), None)
    ncols = []
    for i in range(1, 7):
        for candidate in [f"당번{i}", f"당첨번호{i}", f"n{i}", f"N{i}"]:
            if candidate in df.columns:
                ncols.append(candidate)
                break
    if len(ncols) < 6:
        # Fallback: after draw/date/machine columns
        candidates = [c for c in df.columns if re.search(r"\d", c) and c != draw_col]
        ncols = candidates[:6]
    bonus_col = next((c for c in df.columns if c in ["보볼", "보너스", "bonus", "Bonus"]), None)
    if bonus_col is None:
        bonus_col = next((c for c in df.columns if "보" in c), df.columns[9])
    draws = []
    for _, row in df.iterrows():
        try:
            rnd = _extract_int(row[draw_col])
            date = str(row[date_col]) if date_col else ""
            nums = tuple(sorted(_extract_int(row[c]) for c in ncols[:6]))
            bonus = _extract_int(row[bonus_col])
            draws.append(Draw(rnd, date, nums, bonus, source=str(path)))
        except Exception:
            continue
    return validate_draws(draws)


def merge_draws(*draw_lists: Iterable[Draw]) -> list[Draw]:
    by_round: dict[int, Draw] = {}
    for draws in draw_lists:
        for d in draws:
            by_round[d.round] = d
    return validate_draws(sorted(by_round.values(), key=lambda d: d.round))


def validate_draws(draws: Iterable[Draw]) -> list[Draw]:
    valid = []
    seen = set()
    for d in sorted(draws, key=lambda x: x.round):
        if d.round in seen:
            continue
        if len(d.numbers) != 6 or len(set(d.numbers)) != 6:
            raise ValueError(f"본번호 중복 또는 개수 오류: {d}")
        if not all(1 <= n <= 45 for n in d.numbers):
            raise ValueError(f"본번호 범위 오류: {d}")
        if not 1 <= d.bonus <= 45:
            raise ValueError(f"보너스 범위 오류: {d}")
        if d.bonus in d.numbers:
            raise ValueError(f"보너스와 본번호 중복: {d}")
        seen.add(d.round)
        valid.append(d)
    return valid


def draws_to_frame(draws: list[Draw]) -> pd.DataFrame:
    return pd.DataFrame([d.as_dict() for d in draws])


def load_history(csv_path: str | Path | None = None, xls_path: str | Path | None = None) -> list[Draw]:
    pieces = []
    if csv_path and Path(csv_path).exists():
        pieces.append(load_lotto_csv(csv_path))
    if xls_path and Path(xls_path).exists():
        pieces.append(load_lotto_html_xls(xls_path))
    if not pieces:
        raise FileNotFoundError("CSV/XLS 데이터 파일을 찾지 못했습니다.")
    return merge_draws(*pieces)

```


# FILE: src/engine.py

```py
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

```


# FILE: src/harness/agents.py

```py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import AgentResult, HarnessContext
from ..data_loader import load_history, validate_draws
from ..engine import run_core_pipeline, save_pipeline_outputs
from ..llm_engine import analyze_with_llm


class PlannerAgent:
    name = "planner"
    role = "기획/실험 설계 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        r = AgentResult(self.name, "ok", "실험 계획 수립 완료")
        ctx.artifacts["plan"] = {
            "target_round": ctx.cfg.target_round,
            "n_sets": ctx.cfg.n_sets,
            "backtest_window": ctx.cfg.backtest_window,
            "candidate_samples": ctx.cfg.candidate_samples,
            "quality_gates": [
                "데이터 유효성 오류 0건",
                "미래 데이터 누수 금지",
                "최종 세트 개수 일치",
                "세트 간 overlap 제한 충족",
                "당첨 보장 표현 금지",
            ],
        }
        ctx.log("PlannerAgent: plan created")
        return r.finish()


class DataIngestionAgent:
    name = "data_ingestion"
    role = "데이터 수집/정규화 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        draws = load_history(ctx.csv_path, ctx.xls_path)
        ctx.data["draws"] = draws
        msg = f"{draws[0].round}~{draws[-1].round}회, 총 {len(draws)}개 회차 로드"
        ctx.log(f"DataIngestionAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"round_min": draws[0].round, "round_max": draws[-1].round, "rows": len(draws)}).finish()


class FeatureBacktestAgent:
    name = "feature_backtest"
    role = "전략 구현/백테스트 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        draws = ctx.data["draws"]
        result = run_core_pipeline(draws, ctx.cfg)
        ctx.artifacts["pipeline"] = result
        summary = result["backtest_summary"]
        msg = f"전략 {len(summary)}개 rolling-forward 백테스트 및 후보 생성 완료"
        ctx.log(f"FeatureBacktestAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"strategy_count": len(summary), "candidate_count": len(result["candidates"])}).finish()


class VerificationAgent:
    name = "verification"
    role = "검증/품질 게이트 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts["pipeline"]
        final_sets = result["final_sets"]
        errors = []
        if len(final_sets) != ctx.cfg.n_sets:
            errors.append(f"최종 세트 개수 불일치: {len(final_sets)}")
        for s in final_sets:
            if len(s) != 6 or len(set(s)) != 6 or not all(1 <= n <= 45 for n in s):
                errors.append(f"번호 유효성 오류: {s}")
        for i in range(len(final_sets)):
            for j in range(i + 1, len(final_sets)):
                if len(set(final_sets[i]) & set(final_sets[j])) > ctx.cfg.max_overlap:
                    errors.append(f"overlap 초과: {final_sets[i]} vs {final_sets[j]}")
        status = "ok" if not errors else "error"
        msg = "품질 게이트 통과" if not errors else "; ".join(errors)
        ctx.artifacts["verification"] = {"status": status, "errors": errors}
        ctx.log(f"VerificationAgent: {msg}")
        return AgentResult(self.name, status, msg, {"errors": errors}).finish()


class LlmAnalystAgent:
    name = "llm_analyst"
    role = "LLM 리포트/비판 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts.get("pipeline", {})
        summary = {
            "target_round": ctx.cfg.target_round,
            "final_sets": result.get("final_sets", []),
            "probability": result.get("probability", {}),
            "backtest_summary": result.get("backtest_summary").head(10).to_dict(orient="records") if result.get("backtest_summary") is not None else [],
            "top_candidates": result.get("candidates").head(10).to_dict(orient="records") if result.get("candidates") is not None else [],
        }
        if not ctx.cfg.hf_token:
            msg = "HF_TOKEN 미설정: LLM 분석 생략"
            ctx.artifacts["llm"] = {"ok": False, "error": msg}
            return AgentResult(self.name, "skipped", msg).finish()
        llm = analyze_with_llm(ctx.cfg, summary)
        ctx.artifacts["llm"] = llm
        return AgentResult(self.name, "ok" if llm.get("ok") else "warning", "LLM 분석 완료" if llm.get("ok") else str(llm.get("error"))).finish()


class ReportAgent:
    name = "reporter"
    role = "리포트/산출물 저장 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts["pipeline"]
        paths = save_pipeline_outputs(result, ctx.cfg.output_dir)
        report = {
            "plan": ctx.artifacts.get("plan"),
            "final_sets": result.get("final_sets"),
            "probability": result.get("probability"),
            "verification": ctx.artifacts.get("verification"),
            "agent_results": [a.__dict__ for a in ctx.agent_results],
            "llm": ctx.artifacts.get("llm"),
        }
        p = Path(ctx.cfg.output_dir) / "harness_report.json"
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["harness_report"] = str(p)
        ctx.artifacts["paths"] = paths
        ctx.log("ReportAgent: artifacts saved")
        return AgentResult(self.name, "ok", "리포트 및 CSV/JSON 산출물 저장 완료", paths).finish()

```


# FILE: src/harness/base.py

```py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class AgentResult:
    name: str
    status: str
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str = ""

    def finish(self) -> "AgentResult":
        self.ended_at = datetime.utcnow().isoformat()
        return self


@dataclass
class HarnessContext:
    cfg: Any
    csv_path: str | None = None
    xls_path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.logs.append(f"{datetime.utcnow().isoformat()} | {msg}")


class HarnessAgent(Protocol):
    name: str
    role: str
    def run(self, ctx: HarnessContext) -> AgentResult:
        ...


@dataclass
class AgentSpec:
    name: str
    role: str
    responsibilities: list[str]
    inputs: list[str]
    outputs: list[str]
    quality_gates: list[str]

```


# FILE: src/harness/orchestrator.py

```py
from __future__ import annotations

from .base import HarnessContext, AgentSpec
from .agents import PlannerAgent, DataIngestionAgent, FeatureBacktestAgent, VerificationAgent, LlmAnalystAgent, ReportAgent


def agent_team_spec() -> list[AgentSpec]:
    return [
        AgentSpec("planner", "기획", ["실험 목적 정의", "품질 게이트 설정"], ["config"], ["plan"], ["target_round 확인"]),
        AgentSpec("data_ingestion", "데이터", ["CSV/XLS 파싱", "회차 병합", "유효성 검사"], ["csv", "xls"], ["draws"], ["번호 범위/중복/연속성 검사"]),
        AgentSpec("feature_backtest", "구현/백테스트", ["전략 점수 계산", "rolling-forward 백테스트", "후보 조합 생성"], ["draws"], ["pipeline"], ["미래누수 금지"]),
        AgentSpec("verification", "검증", ["세트 유효성 검사", "overlap 검사", "확률 요약 확인"], ["pipeline"], ["verification"], ["품질 게이트 통과"]),
        AgentSpec("llm_analyst", "AI 분석", ["LLM 보조 해석", "과장 표현 제거"], ["pipeline", "HF settings"], ["llm"], ["당첨 보장 표현 금지"]),
        AgentSpec("reporter", "리포트", ["CSV/JSON 저장", "harness report 생성"], ["pipeline", "verification"], ["outputs"], ["파일 생성 확인"]),
    ]


class AgentHarness:
    def __init__(self, include_llm: bool = False):
        self.agents = [PlannerAgent(), DataIngestionAgent(), FeatureBacktestAgent(), VerificationAgent()]
        if include_llm:
            self.agents.append(LlmAnalystAgent())
        self.agents.append(ReportAgent())

    def run(self, ctx: HarnessContext) -> HarnessContext:
        for agent in self.agents:
            result = agent.run(ctx)
            ctx.agent_results.append(result)
            if result.status == "error":
                ctx.log(f"Harness stopped at {agent.name}: {result.message}")
                break
        return ctx

```


# FILE: src/llm_engine.py

```py
from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import AppConfig


def build_lotto_prompt(result_summary: dict[str, Any]) -> str:
    return (
        "당신은 로또 6/45 통계 연구 보조 분석가입니다. "
        "당첨 보장, 확률 과장, 신비주의 표현을 금지합니다. "
        "아래 JSON의 후보 세트와 백테스트 요약을 근거로 연구용 해석을 작성하세요.\n\n"
        + json.dumps(result_summary, ensure_ascii=False, indent=2)[:12000]
    )


class HfRouterClient:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.url = "https://router.huggingface.co/v1/chat/completions"

    def available(self) -> bool:
        return self.cfg.llm_engine == "hf_api" and bool(self.cfg.hf_token)

    def chat(self, prompt: str) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 또는 LLM_ENGINE 설정이 없습니다."}
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": "You are a rigorous statistical research assistant. Avoid any guaranteed lottery claims."},
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for model in self.cfg.model_candidates:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.cfg.hf_temperature,
                "max_tokens": self.cfg.hf_max_tokens,
                "stream": False,
            }
            for attempt in range(self.cfg.hf_max_retries):
                try:
                    r = requests.post(
                        self.url,
                        headers=headers,
                        json=payload,
                        timeout=(self.cfg.hf_timeout_connect, self.cfg.hf_timeout_read),
                    )
                    if r.status_code < 400:
                        data = r.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        return {"ok": True, "model": model, "content": content, "raw": data}
                    last_error = f"{model}: HTTP {r.status_code} {r.text[:500]}"
                except Exception as e:
                    last_error = f"{model}: {type(e).__name__}: {e}"
                time.sleep(0.8 * (attempt + 1))
        return {"ok": False, "error": last_error or "LLM 호출 실패"}


def analyze_with_llm(cfg: AppConfig, result_summary: dict[str, Any]) -> dict[str, Any]:
    client = HfRouterClient(cfg)
    return client.chat(build_lotto_prompt(result_summary))

```


# FILE: src/metrics.py

```py
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

```


# FILE: src/strategies.py

```py
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

```


# FILE: tests/test_loader.py

```py
from src.data_loader import load_history


def test_load_history_default():
    draws = load_history("data/korealotto_1224V_CSV2.csv", "data/LottoSpecial_공나온순서_1225.xls")
    assert len(draws) >= 1225
    assert draws[-1].round >= 1225
    assert len(draws[-1].numbers) == 6

```

```


## README.md

```md
# Korea Lotto Agent Harness v6.3 HF Gemma4

CLI를 사용하지 않고 **Streamlit 화면에서 HF_TOKEN과 Gemma4 Router 모델을 입력해 실행**하는 버전입니다.

## 핵심 기능

- Streamlit UI에서 HF_TOKEN 입력 또는 Streamlit Secrets 사용
- 기본 모델: `google/gemma-4-26B-A4B-it:deepinfra`
- 후보 모델 fallback 지원
- HF 연결 테스트 버튼
- 통계 엔진: 빈도/최근감쇠/gap/hot-cold/momentum/Bayes/pair affinity/prime factor/anti-crowd
- Agent Harness: 기획 → 데이터 검증 → 백테스트 → 후보 생성 → 검증 → Gemma4 분석 → 리포트 저장
- VS Code 로컬, Streamlit Cloud, Colab 지원

## VS Code에서 Streamlit 실행

```powershell
cd C:\0MyWork1\korea_lotto_agent_harness_v6_3_hf_gemma
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

앱 화면 왼쪽 사이드바에서 `HF_TOKEN`을 입력하거나, Streamlit Secrets에 넣으세요.

## Streamlit Cloud Secrets

```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_새로_발급한_토큰"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = "1400"
HF_TEMPERATURE = "0.2"
HF_TIMEOUT_CONNECT = "10"
HF_TIMEOUT_READ = "120"
HF_MAX_RETRIES = "3"
```

## 주의

Gemma4는 통계 엔진이 만든 후보 세트를 설명하고 비판하는 보조 분석기로 사용됩니다. 공정추첨을 전제로 모든 6개 조합의 실제 1등 확률은 동일합니다.

```


## STREAMLIT_SECRETS_TEMPLATE.toml

```toml
LLM_ENGINE = "hf_api"
HF_TOKEN = "hf_새로_발급한_토큰"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
HF_MAX_TOKENS = "1400"
HF_TEMPERATURE = "0.2"
HF_TIMEOUT_CONNECT = "10"
HF_TIMEOUT_READ = "120"
HF_MAX_RETRIES = "3"

```


## UPDATE_STRATEGY.md

```md
# v6.3 업데이트 전략: CLI → Streamlit + HF Gemma4 중심

## 변경 목표

기존 CLI 실행 흐름을 보조 수단으로 남기되, 실제 사용 흐름은 Streamlit 화면에서 다음을 모두 처리하도록 변경한다.

1. 데이터 업로드
2. 백테스트 파라미터 설정
3. HF_TOKEN 입력 또는 Secrets 사용
4. Gemma4 모델 연결 테스트
5. Agent Harness 실행
6. Gemma4 보조 분석 확인
7. 결과 JSON/CSV 다운로드

## 설계 원칙

- HF_TOKEN은 코드에 하드코딩하지 않는다.
- Streamlit Secrets를 1순위로 사용한다.
- 화면 입력 HF_TOKEN은 현재 세션에서만 사용한다.
- Gemma4는 번호를 직접 생성하는 신비주의 엔진이 아니라, 통계 엔진 결과를 설명·검증·비판하는 LLM Analyst Agent로 쓴다.
- HF Router 모델은 provider suffix를 허용한다. 예: `google/gemma-4-26B-A4B-it:deepinfra`.

## 다음 v6.4 후보

- Gemma4 JSON 구조화 출력
- LLM 재랭커 모드 추가
- 모델별 latency/오류율 로그
- HF 프록시 서버 분리
- 사용량 quota/rate-limit

```


## app.py

```py
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

```


## cli.py

```py
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness


def main():
    ap = argparse.ArgumentParser(description="Korea Lotto 6/45 Agent Harness v6")
    ap.add_argument("--csv", default="data/korealotto_1224V_CSV2.csv")
    ap.add_argument("--xls", default="data/LottoSpecial_공나온순서_1225.xls")
    ap.add_argument("--target", type=int, default=1226)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--backtest", type=int, default=180)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig.from_env(
        target_round=args.target,
        n_sets=args.sets,
        candidate_samples=args.samples,
        backtest_window=args.backtest,
        output_dir=args.outdir,
    )
    ctx = HarnessContext(cfg=cfg, csv_path=args.csv, xls_path=args.xls)
    ctx = AgentHarness(include_llm=args.llm).run(ctx)

    result = ctx.artifacts.get("pipeline", {})
    print(f"\n[1226 후보 {cfg.n_sets}세트]")
    for i, s in enumerate(result.get("final_sets", []), 1):
        print(f"Set {i}: {', '.join(map(str, s))}")
    print("\n[Agent Results]")
    for ar in ctx.agent_results:
        print(f"- {ar.name}: {ar.status} | {ar.message}")
    print(f"\noutputs: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()

```


## docs/AGENT_TEAM_DESIGN.md

```md
# 에이전트 팀 설계

이 프로젝트는 단일 함수형 예측기가 아니라 **Agent Harness**입니다. 각 에이전트는 입력, 출력, 품질 게이트를 갖고 순차적으로 실행됩니다.

## 팀 구성

| 에이전트 | 역할 | 핵심 산출물 |
|---|---|---|
| PlannerAgent | 연구 계획, 품질 게이트 정의 | plan |
| DataIngestionAgent | CSV/XLS 로딩, 정규화, 검증 | draws |
| FeatureBacktestAgent | 전략 구현, rolling-forward 백테스트, 후보 생성 | pipeline |
| VerificationAgent | 최종 세트 유효성, 중복도, 확률 요약 검증 | verification |
| LlmAnalystAgent | HF Router 기반 보조 해석 | llm |
| ReportAgent | CSV/JSON 산출물 저장 | outputs |

## 하네스 철학

- 번호 추천보다 **검증 가능한 실험 루틴**이 우선입니다.
- rolling-forward 외의 무작위 K-fold는 기본 금지합니다.
- LLM은 예측 엔진이 아니라 리포트/비판/설명 계층입니다.
- 당첨 보장 표현은 모든 산출물에서 금지합니다.

```


## docs/HARNESS_SPEC.md

```md
# Harness Specification

## 입력

- `korealotto_1224V_CSV2.csv`: 1~1224회 기본 회차 데이터
- `LottoSpecial_공나온순서_1225.xls`: 1225회 포함 공나온순서 HTML/XLS 파일
- Streamlit 업로드 파일 또는 CLI 경로

## 출력

- `prediction_result.json`
- `harness_report.json`
- `backtest_summary.csv`
- `backtest_detail.csv`
- `weights.csv`
- `number_scores.csv`
- `candidates.csv`

## 품질 게이트

1. 번호 범위 1~45
2. 본번호 6개 중복 없음
3. 보너스와 본번호 중복 없음
4. 회차 병합 후 정렬
5. 백테스트는 t 시점에서 t+1만 예측
6. 최종 세트 수 일치
7. 세트 간 overlap 제한 준수

```


## docs/HF_GEMMA4_INTEGRATION.md

```md
# HF Gemma4 Integration

이 프로젝트는 Hugging Face Router의 OpenAI-compatible Chat Completions API를 사용한다.

기본 엔드포인트:

```text
https://router.huggingface.co/v1/chat/completions
```

기본 모델:

```text
google/gemma-4-26B-A4B-it:deepinfra
```

Fallback 후보:

```text
google/gemma-4-26B-A4B-it:deepinfra
google/gemma-4-26B-A4B-it:novita
google/gemma-4-31B-it:deepinfra
google/gemma-4-31B-it:together
Qwen/Qwen3.5-9B:together
Qwen/Qwen2.5-7B-Instruct:together
```

Gemma4는 `LlmAnalystAgent`에서 실행되며, 통계 엔진의 `final_sets`, `backtest_summary`, `top_candidates`, `probability`만 근거로 분석 리포트를 생성한다.

```


## docs/PRIME_FACTOR_STRATEGY.md

```md
# Prime Factor Strategy v6.2

이 문서는 로또 번호를 소인수분해하여 연구용 예측 feature로 사용하는 전략을 설명합니다.

## 핵심 아이디어

각 번호 `1~45`를 소인수 지수 벡터로 변환합니다. 예를 들면 `12 = 2^2 * 3`, `27 = 3^3`, `38 = 2 * 19`, `45 = 3^2 * 5`입니다. 그 후 회차별 당첨번호 6개를 다음 관점에서 분석합니다.

- 소인수별 장기 출현량
- 소인수별 최근 60회 출현량
- 소인수별 현재 gap / 평균 gap
- prime number count
- composite count
- omega sum: 소인수 지수 총합
- square-free count
- semiprime count
- 2, 3, 5, 7로 나누어지는 숫자 개수

## 구현 위치

- `src/strategies.py`
  - `PrimeFactorStrategy`
  - `prime_factor_exponents`
  - `draw_prime_profile`
  - `prime_combo_structural_score`

- `src/engine.py`
  - 후보 조합 scoring에 `prime_struct_score` 반영
  - 후보 테이블에 `prime_count`, `omega_sum`, `squarefree_count`, `semiprime_count` 출력

## 주의

이 전략은 과거 데이터의 소인수 구조를 feature로 추가하는 연구용 알고리즘입니다. 공정 추첨에서 특정 조합의 물리적 당첨확률을 바꾸지는 않습니다. 신뢰도는 반드시 rolling-forward 백테스트 결과로만 판단해야 합니다.

```


## notebooks/colab_single_file.py

```py
# Google Colab / Jupyter 실행용: 프로젝트 루트 자동 탐색 + src import 오류 방지
!pip -q install pandas numpy matplotlib requests python-dotenv lxml openpyxl streamlit

from pathlib import Path
import zipfile, os, sys, json, shutil

PROJECT_NAME = "korea_lotto_agent_harness_v6"
ZIP_NAMES = [
    "/content/korea_lotto_agent_harness_v6.zip",
    "/content/korea_lotto_agent_harness_v6_1_colab_fix.zip",
    str(Path.cwd() / "korea_lotto_agent_harness_v6.zip"),
    str(Path.cwd() / "korea_lotto_agent_harness_v6_1_colab_fix.zip"),
]

# 1) ZIP이 있으면 압축 해제
zip_found = None
for zp in ZIP_NAMES:
    if Path(zp).exists():
        zip_found = Path(zp)
        break

# Colab에서 ZIP을 아직 업로드하지 않았다면 업로드 창 표시
if zip_found is None and Path('/content').exists():
    try:
        from google.colab import files
        print("ZIP 파일을 업로드하세요: korea_lotto_agent_harness_v6.zip")
        uploaded = files.upload()
        for name in uploaded.keys():
            if name.endswith('.zip'):
                zip_found = Path('/content') / name
                break
    except Exception as e:
        print("Colab files.upload()를 사용할 수 없습니다:", e)

if zip_found is not None:
    print("압축 해제:", zip_found)
    with zipfile.ZipFile(zip_found, "r") as z:
        z.extractall("/content" if Path('/content').exists() else Path.cwd())
else:
    print("ZIP 파일을 찾지 못했습니다. 이미 압축 해제된 폴더를 탐색합니다.")

# 2) src/config.py가 있는 실제 프로젝트 루트 자동 탐색
def find_project_root():
    roots = []
    for base in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if base.exists():
            roots.append(base)
    seen = set()
    for base in roots:
        for p in base.rglob('src/config.py'):
            root = p.parent.parent
            if root not in seen:
                seen.add(root)
                # app.py 또는 cli.py가 있는 프로젝트를 우선 선택
                if (root / 'app.py').exists() or (root / 'cli.py').exists():
                    return root
    return None

PROJECT = find_project_root()
if PROJECT is None:
    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. ZIP을 /content에 업로드했는지, "
        "압축 해제 후 src/config.py가 존재하는지 확인하세요."
    )

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("프로젝트 루트:", PROJECT)
print("현재 작업 폴더:", Path.cwd())
print("src/config.py 존재:", (PROJECT / 'src' / 'config.py').exists())

# 3) 기본 실행
from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness

cfg = AppConfig.from_env(
    target_round=1226,
    n_sets=2,
    candidate_samples=12000,
    backtest_window=180,
    output_dir="/content/lotto_outputs" if Path('/content').exists() else str(PROJECT / 'outputs_colab')
)
ctx = HarnessContext(
    cfg=cfg,
    csv_path=str(PROJECT / "data" / "korealotto_1224V_CSV2.csv"),
    xls_path=str(PROJECT / "data" / "LottoSpecial_공나온순서_1225.xls")
)
ctx = AgentHarness(include_llm=False).run(ctx)

result = ctx.artifacts["pipeline"]
print("\n1226회차 후보 2세트")
for i, s in enumerate(result["final_sets"], 1):
    print(f"Set {i}: {s}")

print("\n에이전트 실행 결과")
for ar in ctx.agent_results:
    print(ar.name, ar.status, ar.message)

print("\n저장 경로:", cfg.output_dir)
print([str(p) for p in Path(cfg.output_dir).glob("*")])
```


## pyproject.toml

```toml
[project]
name = "korea-lotto-agent-harness-v6"
version = "6.0.0"
description = "Korean Lotto 6/45 research harness with agent-team orchestration, rolling backtest, Streamlit, Colab, and HF LLM hooks."
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

```


## requirements.txt

```txt
streamlit>=1.32
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
requests>=2.31
python-dotenv>=1.0
lxml>=4.9
openpyxl>=3.1

```


## src/__init__.py

```py

```


## src/config.py

```py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _streamlit_secret(key: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        return None
    return None


def get_setting(key: str, default: str = "") -> str:
    val = _streamlit_secret(key)
    if val is not None and val != "":
        return val
    return os.getenv(key, default)


@dataclass
class AppConfig:
    target_round: int = 1226
    n_sets: int = 2
    candidate_samples: int = 12000
    backtest_window: int = 180
    min_train_rounds: int = 200
    seed: int = 20260530
    max_overlap: int = 2
    sum_min: int = 98
    sum_max: int = 177
    output_dir: str = "outputs"

    llm_engine: str = "hf_api"
    hf_token: str = ""
    hf_router_model: str = "google/gemma-4-26B-A4B-it:deepinfra"
    hf_model_candidates: str = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
    hf_max_tokens: int = 1400
    hf_temperature: float = 0.2
    hf_timeout_connect: int = 10
    hf_timeout_read: int = 120
    hf_max_retries: int = 3

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppConfig":
        cfg = cls(
            llm_engine=get_setting("LLM_ENGINE", "hf_api"),
            hf_token=get_setting("HF_TOKEN", ""),
            hf_router_model=get_setting("HF_ROUTER_MODEL", get_setting("HF_MODEL_ID", "google/gemma-4-26B-A4B-it:deepinfra")),
            hf_model_candidates=get_setting("HF_MODEL_CANDIDATES", cls.hf_model_candidates),
            hf_max_tokens=int(float(get_setting("HF_MAX_TOKENS", "1400"))),
            hf_temperature=float(get_setting("HF_TEMPERATURE", "0.2")),
            hf_timeout_connect=int(float(get_setting("HF_TIMEOUT_CONNECT", "10"))),
            hf_timeout_read=int(float(get_setting("HF_TIMEOUT_READ", "120"))),
            hf_max_retries=int(float(get_setting("HF_MAX_RETRIES", "3"))),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    @property
    def model_candidates(self) -> list[str]:
        seen = []
        for m in [self.hf_router_model] + [x.strip() for x in self.hf_model_candidates.split(",") if x.strip()]:
            if m not in seen:
                seen.append(m)
        return seen

```


## src/data_loader.py

```py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class Draw:
    round: int
    date: str
    numbers: tuple[int, int, int, int, int, int]
    bonus: int
    source: str = ""

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "date": self.date,
            "n1": self.numbers[0],
            "n2": self.numbers[1],
            "n3": self.numbers[2],
            "n4": self.numbers[3],
            "n5": self.numbers[4],
            "n6": self.numbers[5],
            "bonus": self.bonus,
            "source": self.source,
        }


def _clean_col(c) -> str:
    return str(c).strip().replace(" ", "")


def _extract_int(value) -> int:
    s = str(value).replace(",", "")
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError(f"정수를 추출할 수 없습니다: {value!r}")
    return int(m.group())


def _read_csv_any(path: str | Path) -> pd.DataFrame:
    last = None
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV 로딩 실패: {path}, 마지막 오류={last}")


def load_lotto_csv(path: str | Path) -> list[Draw]:
    df = _read_csv_any(path)
    df.columns = [_clean_col(c) for c in df.columns]
    draw_col = next((c for c in df.columns if c in ["회차", "round", "Round", "draw"]), df.columns[0])
    date_col = next((c for c in df.columns if c in ["추첨일", "당첨일자", "date", "Date"]), None)

    # Korean official-like column names first, then positional fallback.
    ncols = []
    for i in range(1, 7):
        for candidate in [f"당첨번호{i}", f"당번{i}", f"n{i}", f"N{i}"]:
            if candidate in df.columns:
                ncols.append(candidate)
                break
    if len(ncols) < 6:
        numeric_cols = [c for c in df.columns if c not in [draw_col, date_col] and "보" not in c]
        ncols = numeric_cols[:6]
    bonus_col = next((c for c in df.columns if c in ["보너스", "보볼", "bonus", "Bonus"]), None)
    if bonus_col is None:
        candidates = [c for c in df.columns if "보" in c]
        bonus_col = candidates[0] if candidates else df.columns[8]

    draws: list[Draw] = []
    for _, row in df.iterrows():
        try:
            rnd = _extract_int(row[draw_col])
            date = str(row[date_col]) if date_col else ""
            nums = tuple(sorted(_extract_int(row[c]) for c in ncols[:6]))
            bonus = _extract_int(row[bonus_col])
            draws.append(Draw(rnd, date, nums, bonus, source=str(path)))
        except Exception:
            continue
    return validate_draws(draws)


def load_lotto_html_xls(path: str | Path) -> list[Draw]:
    # The uploaded LottoSpecial file is an UTF-8 HTML table with .xls extension.
    tables = pd.read_html(path, encoding="utf-8")
    if not tables:
        return []
    df = tables[0]
    df.columns = [_clean_col(c) for c in df.columns]
    draw_col = next((c for c in df.columns if c in ["회차", "round", "Round"]), df.columns[0])
    date_col = next((c for c in df.columns if c in ["당첨일자", "추첨일", "date", "Date"]), None)
    ncols = []
    for i in range(1, 7):
        for candidate in [f"당번{i}", f"당첨번호{i}", f"n{i}", f"N{i}"]:
            if candidate in df.columns:
                ncols.append(candidate)
                break
    if len(ncols) < 6:
        # Fallback: after draw/date/machine columns
        candidates = [c for c in df.columns if re.search(r"\d", c) and c != draw_col]
        ncols = candidates[:6]
    bonus_col = next((c for c in df.columns if c in ["보볼", "보너스", "bonus", "Bonus"]), None)
    if bonus_col is None:
        bonus_col = next((c for c in df.columns if "보" in c), df.columns[9])
    draws = []
    for _, row in df.iterrows():
        try:
            rnd = _extract_int(row[draw_col])
            date = str(row[date_col]) if date_col else ""
            nums = tuple(sorted(_extract_int(row[c]) for c in ncols[:6]))
            bonus = _extract_int(row[bonus_col])
            draws.append(Draw(rnd, date, nums, bonus, source=str(path)))
        except Exception:
            continue
    return validate_draws(draws)


def merge_draws(*draw_lists: Iterable[Draw]) -> list[Draw]:
    by_round: dict[int, Draw] = {}
    for draws in draw_lists:
        for d in draws:
            by_round[d.round] = d
    return validate_draws(sorted(by_round.values(), key=lambda d: d.round))


def validate_draws(draws: Iterable[Draw]) -> list[Draw]:
    valid = []
    seen = set()
    for d in sorted(draws, key=lambda x: x.round):
        if d.round in seen:
            continue
        if len(d.numbers) != 6 or len(set(d.numbers)) != 6:
            raise ValueError(f"본번호 중복 또는 개수 오류: {d}")
        if not all(1 <= n <= 45 for n in d.numbers):
            raise ValueError(f"본번호 범위 오류: {d}")
        if not 1 <= d.bonus <= 45:
            raise ValueError(f"보너스 범위 오류: {d}")
        if d.bonus in d.numbers:
            raise ValueError(f"보너스와 본번호 중복: {d}")
        seen.add(d.round)
        valid.append(d)
    return valid


def draws_to_frame(draws: list[Draw]) -> pd.DataFrame:
    return pd.DataFrame([d.as_dict() for d in draws])


def load_history(csv_path: str | Path | None = None, xls_path: str | Path | None = None) -> list[Draw]:
    pieces = []
    if csv_path and Path(csv_path).exists():
        pieces.append(load_lotto_csv(csv_path))
    if xls_path and Path(xls_path).exists():
        pieces.append(load_lotto_html_xls(xls_path))
    if not pieces:
        raise FileNotFoundError("CSV/XLS 데이터 파일을 찾지 못했습니다.")
    return merge_draws(*pieces)

```


## src/engine.py

```py
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

```


## src/harness/agents.py

```py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import AgentResult, HarnessContext
from ..data_loader import load_history, validate_draws
from ..engine import run_core_pipeline, save_pipeline_outputs
from ..llm_engine import analyze_with_llm


class PlannerAgent:
    name = "planner"
    role = "기획/실험 설계 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        r = AgentResult(self.name, "ok", "실험 계획 수립 완료")
        ctx.artifacts["plan"] = {
            "target_round": ctx.cfg.target_round,
            "n_sets": ctx.cfg.n_sets,
            "backtest_window": ctx.cfg.backtest_window,
            "candidate_samples": ctx.cfg.candidate_samples,
            "quality_gates": [
                "데이터 유효성 오류 0건",
                "미래 데이터 누수 금지",
                "최종 세트 개수 일치",
                "세트 간 overlap 제한 충족",
                "당첨 보장 표현 금지",
            ],
        }
        ctx.log("PlannerAgent: plan created")
        return r.finish()


class DataIngestionAgent:
    name = "data_ingestion"
    role = "데이터 수집/정규화 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        draws = load_history(ctx.csv_path, ctx.xls_path)
        ctx.data["draws"] = draws
        msg = f"{draws[0].round}~{draws[-1].round}회, 총 {len(draws)}개 회차 로드"
        ctx.log(f"DataIngestionAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"round_min": draws[0].round, "round_max": draws[-1].round, "rows": len(draws)}).finish()


class FeatureBacktestAgent:
    name = "feature_backtest"
    role = "전략 구현/백테스트 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        draws = ctx.data["draws"]
        result = run_core_pipeline(draws, ctx.cfg)
        ctx.artifacts["pipeline"] = result
        summary = result["backtest_summary"]
        msg = f"전략 {len(summary)}개 rolling-forward 백테스트 및 후보 생성 완료"
        ctx.log(f"FeatureBacktestAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"strategy_count": len(summary), "candidate_count": len(result["candidates"])}).finish()


class VerificationAgent:
    name = "verification"
    role = "검증/품질 게이트 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts["pipeline"]
        final_sets = result["final_sets"]
        errors = []
        if len(final_sets) != ctx.cfg.n_sets:
            errors.append(f"최종 세트 개수 불일치: {len(final_sets)}")
        for s in final_sets:
            if len(s) != 6 or len(set(s)) != 6 or not all(1 <= n <= 45 for n in s):
                errors.append(f"번호 유효성 오류: {s}")
        for i in range(len(final_sets)):
            for j in range(i + 1, len(final_sets)):
                if len(set(final_sets[i]) & set(final_sets[j])) > ctx.cfg.max_overlap:
                    errors.append(f"overlap 초과: {final_sets[i]} vs {final_sets[j]}")
        status = "ok" if not errors else "error"
        msg = "품질 게이트 통과" if not errors else "; ".join(errors)
        ctx.artifacts["verification"] = {"status": status, "errors": errors}
        ctx.log(f"VerificationAgent: {msg}")
        return AgentResult(self.name, status, msg, {"errors": errors}).finish()


class LlmAnalystAgent:
    name = "llm_analyst"
    role = "LLM 리포트/비판 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts.get("pipeline", {})
        summary = {
            "target_round": ctx.cfg.target_round,
            "final_sets": result.get("final_sets", []),
            "probability": result.get("probability", {}),
            "backtest_summary": result.get("backtest_summary").head(10).to_dict(orient="records") if result.get("backtest_summary") is not None else [],
            "top_candidates": result.get("candidates").head(10).to_dict(orient="records") if result.get("candidates") is not None else [],
        }
        if not ctx.cfg.hf_token:
            msg = "HF_TOKEN 미설정: LLM 분석 생략"
            ctx.artifacts["llm"] = {"ok": False, "error": msg}
            return AgentResult(self.name, "skipped", msg).finish()
        llm = analyze_with_llm(ctx.cfg, summary)
        ctx.artifacts["llm"] = llm
        return AgentResult(self.name, "ok" if llm.get("ok") else "warning", "LLM 분석 완료" if llm.get("ok") else str(llm.get("error"))).finish()


class ReportAgent:
    name = "reporter"
    role = "리포트/산출물 저장 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts["pipeline"]
        paths = save_pipeline_outputs(result, ctx.cfg.output_dir)
        report = {
            "plan": ctx.artifacts.get("plan"),
            "final_sets": result.get("final_sets"),
            "probability": result.get("probability"),
            "verification": ctx.artifacts.get("verification"),
            "agent_results": [a.__dict__ for a in ctx.agent_results],
            "llm": ctx.artifacts.get("llm"),
        }
        p = Path(ctx.cfg.output_dir) / "harness_report.json"
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["harness_report"] = str(p)
        ctx.artifacts["paths"] = paths
        ctx.log("ReportAgent: artifacts saved")
        return AgentResult(self.name, "ok", "리포트 및 CSV/JSON 산출물 저장 완료", paths).finish()

```


## src/harness/base.py

```py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class AgentResult:
    name: str
    status: str
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str = ""

    def finish(self) -> "AgentResult":
        self.ended_at = datetime.utcnow().isoformat()
        return self


@dataclass
class HarnessContext:
    cfg: Any
    csv_path: str | None = None
    xls_path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.logs.append(f"{datetime.utcnow().isoformat()} | {msg}")


class HarnessAgent(Protocol):
    name: str
    role: str
    def run(self, ctx: HarnessContext) -> AgentResult:
        ...


@dataclass
class AgentSpec:
    name: str
    role: str
    responsibilities: list[str]
    inputs: list[str]
    outputs: list[str]
    quality_gates: list[str]

```


## src/harness/orchestrator.py

```py
from __future__ import annotations

from .base import HarnessContext, AgentSpec
from .agents import PlannerAgent, DataIngestionAgent, FeatureBacktestAgent, VerificationAgent, LlmAnalystAgent, ReportAgent


def agent_team_spec() -> list[AgentSpec]:
    return [
        AgentSpec("planner", "기획", ["실험 목적 정의", "품질 게이트 설정"], ["config"], ["plan"], ["target_round 확인"]),
        AgentSpec("data_ingestion", "데이터", ["CSV/XLS 파싱", "회차 병합", "유효성 검사"], ["csv", "xls"], ["draws"], ["번호 범위/중복/연속성 검사"]),
        AgentSpec("feature_backtest", "구현/백테스트", ["전략 점수 계산", "rolling-forward 백테스트", "후보 조합 생성"], ["draws"], ["pipeline"], ["미래누수 금지"]),
        AgentSpec("verification", "검증", ["세트 유효성 검사", "overlap 검사", "확률 요약 확인"], ["pipeline"], ["verification"], ["품질 게이트 통과"]),
        AgentSpec("llm_analyst", "AI 분석", ["LLM 보조 해석", "과장 표현 제거"], ["pipeline", "HF settings"], ["llm"], ["당첨 보장 표현 금지"]),
        AgentSpec("reporter", "리포트", ["CSV/JSON 저장", "harness report 생성"], ["pipeline", "verification"], ["outputs"], ["파일 생성 확인"]),
    ]


class AgentHarness:
    def __init__(self, include_llm: bool = False):
        self.agents = [PlannerAgent(), DataIngestionAgent(), FeatureBacktestAgent(), VerificationAgent()]
        if include_llm:
            self.agents.append(LlmAnalystAgent())
        self.agents.append(ReportAgent())

    def run(self, ctx: HarnessContext) -> HarnessContext:
        for agent in self.agents:
            result = agent.run(ctx)
            ctx.agent_results.append(result)
            if result.status == "error":
                ctx.log(f"Harness stopped at {agent.name}: {result.message}")
                break
        return ctx

```


## src/llm_engine.py

```py
from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import AppConfig


def build_lotto_prompt(result_summary: dict[str, Any]) -> str:
    return (
        "당신은 로또 6/45 통계 연구 보조 분석가입니다. "
        "당첨 보장, 확률 과장, 신비주의 표현, 99% 적중 같은 표현을 절대 쓰지 마세요. "
        "아래 JSON의 통계 엔진 결과, 후보 세트, 백테스트 요약, 후보 조합 구조만 근거로 "
        "1) 최종 세트 해석, 2) 약점/리스크, 3) 다음 실험 개선안, 4) 과적합 경고를 한국어로 작성하세요. "
        "번호를 새로 조작하지 말고, 주어진 final_sets를 중심으로 설명하세요.\n\n"
        + json.dumps(result_summary, ensure_ascii=False, indent=2)[:16000]
    )


class HfRouterClient:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.url = "https://router.huggingface.co/v1/chat/completions"

    def available(self) -> bool:
        return self.cfg.llm_engine == "hf_api" and bool(self.cfg.hf_token)

    def _post_chat(self, model: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.hf_temperature,
            "max_tokens": int(max_tokens or self.cfg.hf_max_tokens),
            "stream": False,
        }
        r = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=(self.cfg.hf_timeout_connect, self.cfg.hf_timeout_read),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "model": model, "content": content, "raw": data}

    def test_connection(self) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 또는 LLM_ENGINE 설정이 없습니다."}
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "연결 테스트입니다. 한국어로 'HF Router 연결 성공'이라고만 답하세요."},
        ]
        last_error = None
        for model in self.cfg.model_candidates:
            try:
                return self._post_chat(model, messages, max_tokens=80)
            except Exception as e:
                last_error = f"{model}: {type(e).__name__}: {e}"
                time.sleep(0.5)
        return {"ok": False, "error": last_error or "HF 연결 테스트 실패"}

    def chat(self, prompt: str) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 또는 LLM_ENGINE 설정이 없습니다."}
        messages = [
            {"role": "system", "content": "You are a rigorous statistical research assistant. Avoid any guaranteed lottery claims."},
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for model in self.cfg.model_candidates:
            for attempt in range(self.cfg.hf_max_retries):
                try:
                    return self._post_chat(model, messages)
                except Exception as e:
                    last_error = f"{model}: {type(e).__name__}: {e}"
                    time.sleep(0.8 * (attempt + 1))
        return {"ok": False, "error": last_error or "LLM 호출 실패"}


def analyze_with_llm(cfg: AppConfig, result_summary: dict[str, Any]) -> dict[str, Any]:
    client = HfRouterClient(cfg)
    return client.chat(build_lotto_prompt(result_summary))

```


## src/metrics.py

```py
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

```


## src/strategies.py

```py
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

```


## tests/test_loader.py

```py
from src.data_loader import load_history


def test_load_history_default():
    draws = load_history("data/korealotto_1224V_CSV2.csv", "data/LottoSpecial_공나온순서_1225.xls")
    assert len(draws) >= 1225
    assert draws[-1].round >= 1225
    assert len(draws[-1].numbers) == 6

```
