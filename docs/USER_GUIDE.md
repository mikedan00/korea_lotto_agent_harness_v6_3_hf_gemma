# Korea Lotto Agent Harness v7.0 — 실행 방법 & 사용 설명서

> 이 도구는 **통계 연구용**입니다. 공정 추첨에서는 모든 6개 조합의 1등 확률이 1/8,145,060 으로 같습니다.
> 하네스는 각 전략이 무작위보다 나은지 **측정**하고, 우위가 확인되지 않으면 그 사실을 그대로 보여줍니다.

## 목차

1. [준비물](#1-준비물)
2. [설치 (최초 1회)](#2-설치-최초-1회)
3. [실행 방법 4가지](#3-실행-방법-4가지)
4. [Streamlit 화면 사용법](#4-streamlit-화면-사용법)
5. [AI 리뷰 모델 설정](#5-ai-리뷰-모델-설정)
6. [CLI 사용법](#6-cli-사용법)
7. [결과 읽는 법](#7-결과-읽는-법)
8. [Streamlit Cloud 배포](#8-streamlit-cloud-배포)
9. [데이터 업데이트](#9-데이터-업데이트)
10. [문제 해결](#10-문제-해결)

---

## 1. 준비물

| 항목 | 필수 여부 | 비고 |
|---|---|---|
| Python 3.10 이상 | 필수 | `python --version` 으로 확인 |
| VS Code + Python 확장 | 권장 | F5 실행/디버그 구성 제공 |
| Hugging Face 토큰 (`hf_...`) | HF 모델을 쓸 때 | https://huggingface.co/settings/tokens 에서 발급 |
| Node.js 18+ 및 Claude Code CLI | Claude 모델을 쓸 때 (로컬 전용) | 아래 5-2 참고 |

LLM 없이도 통계 엔진·최적화·통계 감사는 모두 동작합니다.

---

## 2. 설치 (최초 1회)

### Windows (PowerShell)

```powershell
cd C:\0MyWork1\korea_lotto_agent_harness_v6_3_hf_gemma
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env
```

> PowerShell 에서 `activate` 가 막히면: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### macOS / Linux

```bash
cd korea_lotto_agent_harness_v6_3_hf_gemma
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

### `.env` 편집

```ini
LLM_PROVIDER=hf_api          # hf_api | claude_cli | none
HF_TOKEN=hf_새로_발급한_토큰   # HF 를 쓸 때만
HF_ROUTER_MODEL=google/gemma-4-26B-A4B-it:deepinfra
CLAUDE_MODEL=sonnet          # claude_cli 를 쓸 때: sonnet / opus / haiku / fable / 전체 모델 ID
CLAUDE_TIMEOUT=240
```

`.env` 는 `.gitignore` 에 포함되어 있어 커밋되지 않습니다. **토큰을 코드나 README 에 직접 쓰지 마세요.**

### 설치 확인

```bash
python -m pytest -q          # 12 passed 가 나오면 정상
python cli.py --list-models  # 모델 프리셋 + Claude CLI 감지 결과
```

---

## 3. 실행 방법 4가지

### A. Streamlit 웹 화면 (권장)

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 이 자동으로 열립니다.

### B. VS Code 에서 F5

1. VS Code 로 프로젝트 폴더 열기
2. `Ctrl+Shift+P` → **Python: Select Interpreter** → `venv` 선택
3. 왼쪽 **실행 및 디버그(Ctrl+Shift+D)** 에서 구성 선택 후 F5

| 구성 이름 | 동작 |
|---|---|
| Streamlit: app.py | 디버거가 붙은 Streamlit 실행 (중단점 사용 가능) |
| CLI: 통계 엔진만 (LLM 없음) | `python cli.py --provider none` |
| CLI: AI 리뷰 - Claude Code CLI | 실행 시 **모델 선택 팝업** → Claude CLI 로 AI 리뷰 |
| CLI: AI 리뷰 - HF Router | 실행 시 **HF 모델 선택 팝업** → `.env` 의 HF_TOKEN 사용 |
| Pytest | 테스트 실행 |

`Ctrl+Shift+P` → **Tasks: Run Task** 로 `의존성 설치`, `Streamlit 실행`, `모델 목록`, `테스트` 작업도 실행할 수 있습니다.

### C. 터미널 CLI

```bash
python cli.py                       # 통계 엔진 + 통계 감사 (LLM 없음, 약 5초)
python cli.py --llm --provider claude_cli --claude-model sonnet
```

자세한 옵션은 [6. CLI 사용법](#6-cli-사용법) 참고.

### D. Google Colab (셀 하나)

설치 없이 브라우저에서 실행합니다. 프로그램 전체와 기본 데이터(1~1224회)가 셀 하나에 들어 있습니다.

1. https://colab.research.google.com 에서 새 노트북을 만듭니다.
2. `notebooks/colab_one_cell.py` 내용을 **전부 복사해 첫 셀에 붙여넣고** 실행(▶)합니다.
   - 또는 `notebooks/korea_lotto_v7_colab.ipynb` 를 Colab 에서 **파일 → 노트북 업로드**로 엽니다.
3. 셀 오른쪽 폼에서 목표 회차, 세트 수, 백테스트 회차, LLM 공급자 등을 바꿀 수 있습니다.
4. `UPLOAD_DATA` 가 켜져 있으면 실행 중 업로드 창이 뜹니다.
   - 공나온순서 XLS/HTML 을 올리면 기본 CSV 와 회차 기준으로 합쳐집니다(같은 회차는 XLS 값 우선).
   - 전체 회차 CSV 를 올리면 기본 CSV 대신 사용합니다.
   - 추가 데이터가 없으면 **취소**를 누르면 됩니다.
5. 결과는 셀 아래에 표로 표시되고, `lotto_results.zip` 으로 내려받아집니다.

AI 리뷰를 쓰려면 Colab 왼쪽 🔑 **보안 비밀**에 키를 저장하고 **노트북 액세스**를 켜세요.

| `LLM_PROVIDER` | 필요한 보안 비밀 | 비고 |
|---|---|---|
| `none` | — | 통계 엔진과 통계 감사만 실행 (기본값, 약 10초) |
| `hf_api` | `HF_TOKEN` | `HF_MODEL` 에 모델 입력 |
| `claude_cli` | `ANTHROPIC_API_KEY` | Claude Code CLI 가 없으면 셀이 `npm` 으로 자동 설치. `CLAUDE_MODEL` 선택 |

`APPLY_AI_SUGGESTIONS` 를 켜면 AI 자문 제안을 적용해 한 번 더 실행하고 두 결과를 차례로 보여줍니다.

> 개발자용: `src/` 를 고친 뒤에는 `python tools/build_colab_cell.py` 로 셀을 다시 생성하세요. 테스트가 셀이 최신인지 검사합니다.

---

## 4. Streamlit 화면 사용법

### 사이드바

**① 실행 설정**

| 항목 | 기본값 | 설명 |
|---|---|---|
| 목표 회차 | 1226 | 리포트에 표시할 회차 번호 |
| 최종 세트 수 | 2 | 뽑을 6개 번호 세트 개수 (1~10) |
| 후보 조합 샘플 수 | 12000 | 클수록 후보가 다양하지만 느려짐 |
| walk-forward 백테스트 회차 | 180 | 최근 몇 회차로 전략을 검증할지. 클수록 통계가 안정적이지만 느려짐 |
| 세트 간 최대 중복 | 2 | 서로 다른 세트가 공유할 수 있는 번호 수 |

**② 최적화 고급 설정** (펼쳐서 사용)

| 항목 | 기본값 | 설명 |
|---|---|---|
| 앙상블 가중치·온도 최적화 | 켬 | 끄면 14개 전략을 같은 비중으로 사용 |
| tune 비율 | 0.7 | 백테스트 앞 70% 로 가중치를 찾고, 뒤 30% 로 검증 |
| 학습형 구조 필터 coverage | 0.95 | 과거 당첨조합의 95% 가 통과하는 범위만 허용 |
| 후보 탐색 온도 | 0.35 | 0 이면 완전 무작위 탐색, 클수록 상위 점수 번호 위주 |
| 세트 간 커버리지 보너스 | 0.3 | 세트끼리 다른 번호를 쓰도록 유도하는 강도 |
| 후보 점수 가중치 | 1.0 / 0.25 / 0.35 / 0.15 | 앙상블 확률 / pair lift / 구조 전형성 / 소인수 구조 |

**③ 데이터** — 비워두면 `data/korealotto_1224V_CSV2.csv` 사용. 새 회차가 포함된 CSV 또는 공나온순서 XLS/HTML 을 올리면 회차 기준으로 병합됩니다.

**④ AI 리뷰 모델** — [5장](#5-ai-리뷰-모델-설정) 참고.

### 실행 버튼

- **🚀 Agent Harness 실행**: 7개 에이전트가 순서대로 실행되며 진행 상황이 상태 상자에 표시됩니다.
  `planner → data_ingestion → feature_backtest → verification → statistical_audit → ai_review_board → reporter`
- **🤖 AI 자문 제안 적용 후 재실행**: AI 리뷰가 파라미터 조정을 제안했을 때만 활성화됩니다. 제안값(허용 범위로 제한됨)을 적용해 다시 실행하고, 적용된 값은 "최종 결과" 탭 상단에 표시됩니다.

### 결과 탭

| 탭 | 내용 |
|---|---|
| 최종 결과 | 최종 세트, holdout edge / 평균 적중 / 적중 p(단측·양측) / 감사 판정과 95% 신뢰구간, 최종 세트 상세(점수 성분 분해·구조 지표), 에이전트 실행 로그 |
| 최적화·감사 | 통계 감사 경고 목록, holdout 검증표(최적화 앙상블 vs 균등 가중치 기준선), 전략별 가중치 막대그래프, 최적화 상세, 학습된 구조 필터 범위 |
| AI 리뷰 | 분석가 리포트, 비평가 판정(JSON), 최적화 자문(JSON), 적용 가능한 제안, Claude CLI 비용 |
| 백테스트 | 전략별 요약표, 전략별 30회 이동평균 적중 그래프 |
| 데이터/후보 | 번호별 전략 점수·확률, 상위 200개 후보 조합과 구조 지표 |
| 하네스 설계/다운로드 | 에이전트 팀 명세, 결과 CSV/JSON 다운로드 버튼 |

---

## 5. AI 리뷰 모델 설정

AI 리뷰 보드는 같은 모델을 3번 호출합니다.

1. **분석가**: 최종 세트와 백테스트 결과를 해석하는 마크다운 리포트
2. **비평가**: 분석가 리포트의 과장·오류·과적합을 검토해 `accept / revise / reject` 판정 (JSON)
3. **최적화 자문**: 다음 실행 파라미터 제안 (JSON). 허용된 키·범위만 반영됩니다.

AI 는 번호를 직접 만들거나 바꾸지 않습니다.

### 5-1. Hugging Face Router

1. 사이드바 **LLM 공급자 → Hugging Face Router**
2. **HF_TOKEN** 입력 (비워두면 `.env` 또는 Streamlit Secrets 값 사용, 입력값은 현재 세션에서만 사용)
3. **HF 모델** 드롭다운에서 선택하거나 `✏️ 직접 입력` 으로 `조직/모델:프로바이더` 형식 입력
4. **Fallback 후보**: 1순위 모델이 실패하면 순서대로 시도할 모델 목록 (쉼표 구분)
5. **🔌 모델 연결 테스트** 로 확인

### 5-2. Claude Code CLI (로컬 노트북 전용)

설치 및 로그인 (최초 1회):

```bash
npm install -g @anthropic-ai/claude-code
claude          # 브라우저 로그인 또는 API 키 설정 후 종료(/exit)
claude --version
```

사용:

1. Streamlit 을 **같은 터미널 환경**(PATH 에 `claude` 가 있는 상태)에서 실행
2. 사이드바 **LLM 공급자 → Claude Code CLI (로컬)** — 초록색 "감지" 메시지 확인
3. **Claude 모델** 선택: `sonnet`(균형, 기본) / `opus`(고성능) / `haiku`(빠르고 저렴) / `fable` 또는 전체 모델 ID
4. **CLI 타임아웃**: 3단계 리뷰 각각의 제한 시간(기본 240초)
5. **🔌 모델 연결 테스트** → "Claude CLI 연결 성공"

`claude` 가 PATH 에 없으면 `.env` 에 `CLAUDE_CLI_PATH` 로 실행 파일 경로를 지정하세요.
하네스는 Claude CLI 를 **모든 도구를 끈 상태**(`--tools ""`)로 호출하므로 파일을 읽거나 명령을 실행하지 않습니다.
사용 요금은 로그인한 Claude 계정 기준으로 청구되며, 호출 비용은 "AI 리뷰" 탭 하단에 표시됩니다.

### 5-3. LLM 사용 안 함

**LLM 사용 안 함** 을 선택하면 AI 리뷰를 건너뜁니다. 통계 감사는 그대로 실행됩니다.

---

## 6. CLI 사용법

```bash
python cli.py [옵션]
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--csv PATH` | `data/korealotto_1224V_CSV2.csv` | 기본 회차 CSV |
| `--xls PATH` | `data/LottoSpecial_공나온순서_1225.xls` | 추가 회차 XLS/HTML (없으면 무시) |
| `--target N` | 1226 | 목표 회차 |
| `--sets N` | 2 | 최종 세트 수 |
| `--samples N` | 12000 | 후보 조합 샘플 수 |
| `--backtest N` | 180 | walk-forward 백테스트 회차 |
| `--outdir DIR` | `outputs` | 결과 저장 폴더 |
| `--no-optimize` | — | 앙상블 최적화 끄기 (균등 가중치) |
| `--llm` | — | AI 리뷰 보드 실행 |
| `--provider` | `.env` 의 `LLM_PROVIDER` | `hf_api` / `claude_cli` / `none` |
| `--hf-model` | `.env` 값 | HF 1순위 모델 |
| `--claude-model` | `.env` 값 (`sonnet`) | Claude CLI 모델 |
| `--list-models` | — | 프리셋 모델 목록과 Claude CLI 감지 결과만 출력 |

예시:

```bash
python cli.py --list-models
python cli.py --sets 5 --backtest 300 --samples 30000
python cli.py --llm --provider claude_cli --claude-model haiku
python cli.py --llm --provider hf_api --hf-model "Qwen/Qwen3.5-9B:together"
```

AI 리뷰를 켜면 모델 호출 3회 때문에 실행에 수 분이 걸릴 수 있습니다.

---

## 7. 결과 읽는 법

### 핵심 지표

| 지표 | 의미 | 해석 |
|---|---|---|
| **holdout edge (nats)** | 튜닝에 쓰지 않은 검증 구간에서 앙상블 확률의 로그우도 − 균등 분포 로그우도. 95% 신뢰구간(부트스트랩) 함께 표시 | 신뢰구간 하한이 **0보다 커야** 우위가 확인된 것 |
| **holdout 평균 적중** | 검증 구간에서 상위 6개 번호가 실제 당첨번호와 맞은 평균 개수(95% 신뢰구간 포함) | 무작위 기대값은 **0.8개** |
| **holdout 적중 p (단측/양측)** | 단측: 0.8보다 클 확률 검정, 양측: 0.8과 다른지 검정. 검증 구간만 사용 | 단측 p가 **0.05 미만**이어야 무작위보다 낫다는 근거 |
| **균등 가중치 기준선** | 14개 전략을 같은 비중으로 쓴 앙상블을 같은 검증 구간에서 평가한 값 | 최적화 앙상블과의 차이 신뢰구간이 0을 포함하지 않아야 최적화가 의미 있음 |
| **감사 판정** | `signal` / `weak_signal` / `no_evidence_of_edge` | 아래 표 참고 |
| **shrunk_to_uniform** | holdout 에서 실패해 신호를 약하게 만들었는지 | `True` 면 결과가 사실상 무작위 선택에 가까움 |

| 판정 | 조건 |
|---|---|
| `signal` | holdout edge 95% 신뢰구간 하한 > 0, holdout 적중 단측 p ≤ 0.05, Bonferroni 보정 후 유의한 전략 존재 |
| `weak_signal` | holdout edge > 0 이지만 통계적으로 확인되지 않음 |
| `no_evidence_of_edge` | holdout edge ≤ 0 — 무작위 대비 우위 근거 없음 |

1~1224회 데이터의 현재 결과는 `no_evidence_of_edge` 입니다. 공정한 추첨에서 예상되는 결과입니다.

### 백테스트 요약표 컬럼

| 컬럼 | 의미 |
|---|---|
| `mean_hits` | 전략 단독 상위 6개 번호의 평균 적중 |
| `hits_vs_random` | `mean_hits − 0.8` |
| `ll_edge` | 전략 단독 로그우도 edge (음수면 균등보다 나쁨) |
| `z_score`, `p_value` | 무작위 대비 단측 검정 |
| `p_two_sided` | 무작위 대비 양측 검정 |
| `p_bonferroni` | 14개 전략 다중비교 보정 p-value |

### 후보 점수 성분

`score = c_prob + c_pair + c_typical + c_prime` 입니다. 각 성분은 (점수 가중치 × 후보 집합 안에서 표준화한 z-score)이며, 네 값을 더하면 `score` 와 정확히 같습니다. 최종 세트 상세 표의 `score_rank` 는 전체 후보 중 점수 순위입니다. 커버리지 보너스 때문에 2번째 이후 세트는 점수 순위가 낮을 수 있습니다.

### 후보 조합 구조 지표

`sum`(합계), `odd`(홀수 개수), `high`(23 이상 개수), `consecutive_pairs`(연속번호 쌍), `last_overlap`(직전 회차와 겹친 수),
`ac`(AC값: 서로 다른 차이 개수 − 5), `zones_used`(사용한 번호대 수), `max_same_last_digit`(같은 끝수 최대 개수), `prime_count`(소수 개수)

### 저장 파일 (`outputs/`)

| 파일 | 내용 |
|---|---|
| `prediction_result.json` | 최종 세트, 최적화 결과, 필터, 상위 20개 후보 |
| `harness_report.json` | 계획, 검증, 통계 감사, 에이전트 결과, AI 리뷰 전체 |
| `backtest_summary.csv` / `backtest_detail.csv` | 전략별 요약 / 회차별 예측·정답·적중 |
| `weights.csv` | 최적화된 전략 가중치 |
| `number_scores.csv` | 번호별 전략 점수와 앙상블 확률 |
| `candidates.csv` | 전체 후보 조합과 점수 |
| `history.csv` | 로드된 회차 데이터 |

---

## 8. Streamlit Cloud 배포

1. 코드를 GitHub 저장소에 push
2. https://share.streamlit.io → **New app** → 저장소/브랜치 선택, Main file: `app.py`
3. **Advanced settings → Secrets** 에 `STREAMLIT_SECRETS_TEMPLATE.toml` 내용을 붙여넣고 `HF_TOKEN` 을 실제 토큰으로 교체
4. Deploy

주의:

- Streamlit Cloud 에는 Claude Code CLI 가 없으므로 사이드바에 "미감지"로 표시됩니다. 클라우드에서는 **Hugging Face Router** 를 사용하세요.
- 로컬에서 Secrets 를 시험하려면 `.streamlit/secrets.toml.example` 을 `.streamlit/secrets.toml` 로 복사해 편집합니다 (커밋되지 않음).
- 설정 우선순위: **화면 입력값 → Streamlit Secrets → `.env`/환경변수 → 기본값**

---

## 9. 데이터 업데이트

- 새 회차가 추가된 CSV 를 `data/` 에 넣고 `--csv` 로 지정하거나, Streamlit 에서 업로드합니다.
- CSV 컬럼: `회차, 추첨일, 당첨번호1~6, 보너스` (한글/영문 컬럼명, UTF-8/CP949 인코딩 자동 인식)
- 동행복권 공나온순서 XLS(HTML 표 형식)도 업로드할 수 있으며, 같은 회차는 나중에 읽은 파일 값으로 덮어씁니다.
- 로딩 시 번호 범위(1~45), 본번호 중복, 보너스 중복을 검사하며 오류가 있으면 실행이 중단됩니다.

---

## 10. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | 프로젝트 루트 폴더에서 실행하세요 (`cd` 확인) |
| `streamlit: command not found` | 가상환경 활성화 확인, 또는 `python -m streamlit run app.py` |
| HF 연결 테스트 `HTTP 401` | 토큰 오류/만료 → 새로 발급 |
| HF `HTTP 404` / `model not supported` | 해당 프로바이더에서 모델 미지원 → 다른 모델이나 `:프로바이더` 선택, Fallback 후보 확인 |
| HF `HTTP 429` | 사용량 제한 → 잠시 후 재시도 또는 다른 모델 |
| Claude CLI "미감지" | `claude --version` 확인, Streamlit 을 같은 터미널에서 실행, 또는 `.env` 에 `CLAUDE_CLI_PATH` 지정 |
| Claude CLI 오류 (인증) | 터미널에서 `claude` 를 실행해 다시 로그인 |
| Claude CLI 타임아웃 | 사이드바 **CLI 타임아웃** 을 늘리거나 `haiku` 사용 |
| 비평가/자문 결과가 `raw` 텍스트로만 표시 | 모델이 JSON 을 내지 못함 → 더 큰 모델 사용 |
| 실행이 느림 | `walk-forward 백테스트 회차`, `후보 조합 샘플 수` 줄이기 |
| 최종 세트가 요청보다 적음 | `세트 간 최대 중복` 을 늘리거나 샘플 수 늘리기 |
| 실행할 때마다 세트가 같음 | 의도된 동작 (고정 seed). 데이터나 설정이 바뀌면 결과도 바뀝니다 |
