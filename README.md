# Korea Lotto Agent Harness v7.0

로또 6/45 통계 연구 하네스. 지금까지 연구한 전략을 **14개 walk-forward 전략 + 앙상블 최적화기**로 통합하고,
**결정론적 통계 감사 + AI 리뷰 보드(분석가 → 비평가 → 최적화 자문)** 를 붙였습니다.
AI 모델은 **Hugging Face Router**(Streamlit Cloud/로컬) 또는 **Claude Code CLI**(로컬 노트북/VS Code)에서 선택합니다.

> ⚠️ 공정추첨에서 모든 6개 조합의 1등 확률은 1/8,145,060 으로 동일합니다. 이 프로젝트는 "예측률을 높였다"고 주장하지 않고,
> 각 전략이 **무작위 대비 우위가 있는지 정직하게 측정**합니다. 1~1224회 데이터 기준 현재 판정은 `no_evidence_of_edge`
> (holdout edge −0.049, 95% CI −0.077~−0.019, holdout 적중 단측 p=0.97)이며, 이 경우 시스템은 신호를 자동으로 수축합니다.

## v7.0 핵심

| 영역 | 내용 |
|---|---|
| 전략 14개 | long_frequency, recent_decay, gap, hot_cold, momentum, bayes, pair_affinity, prime_factor, anti_crowd + **markov_transition, multi_scale_freq, last_digit, zone_balance, bonus_carry** |
| 최적화 | fold별 점수 캐싱 → 앙상블 확률 `q = softmax(β·Σ w_k z_k)` 의 로그우도 최대화(랜덤 Dirichlet 탐색 + hill-climb, L2 정규화), tune/holdout 분리 검증, holdout 실패 시 β 수축 |
| 검정 | 전략별 top-6 적중 z-검정(단측·양측, 기대 0.8) + Bonferroni 보정, 로그우도 edge(nats). holdout 전용 평가: 부트스트랩 95% CI, 균등 가중치 기준선과 짝지은 비교 |
| 후보 생성 | Gumbel top-k 샘플링, 과거 당첨조합 95% 를 포함하는 **학습형 구조 필터**(합계·홀짝·고저·연속·직전중복·AC·구간수·끝수·소수), 확률/pair lift/전형성/소인수 구조 다성분 점수 |
| 세트 선택 | 점수 + 신규 번호 커버리지 보너스 greedy, 세트 간 overlap 제한 |
| 감사 | `statistical_audit` (LLM 불필요): 과적합 격차, 유의성, 가중치 집중, 필터 통과율 |
| AI 리뷰 | 분석가(마크다운) → 비평가(JSON verdict/risk_flags) → 자문(JSON 파라미터 제안, 허용범위 클램프) → Streamlit 에서 **"AI 자문 제안 적용 후 재실행"** |
| 모델 선택 | HF: 프리셋 드롭다운 + 직접입력 + fallback 목록 / Claude CLI: sonnet·opus·haiku·fable 또는 전체 모델 ID |

📘 **실행 방법 & 사용 설명서: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)** (웹 페이지 버전: `docs/user_guide.html` 을 브라우저로 열기)  
자세한 알고리즘: [docs/V7_ALGORITHMS.md](docs/V7_ALGORITHMS.md)

## 1. 로컬 노트북 (VS Code)

```powershell
cd C:\0MyWork1\korea_lotto_agent_harness_v6_3_hf_gemma
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env   # HF_TOKEN / LLM_PROVIDER / CLAUDE_MODEL 편집
streamlit run app.py
```

VS Code 에서는 `.vscode/launch.json` 의 실행 구성을 사용하세요 (F5).

- **Streamlit: app.py** — 디버거 연결된 Streamlit
- **CLI: AI 리뷰 - Claude Code CLI** — 실행 시 모델 선택 팝업(sonnet/opus/haiku…)
- **CLI: AI 리뷰 - HF Router** — 실행 시 HF 모델 선택 팝업
- **Pytest**

### Claude Code CLI 사용 (로컬 전용)

```bash
npm install -g @anthropic-ai/claude-code   # 또는 공식 설치 스크립트
claude                                      # 최초 1회 로그인
python cli.py --list-models                 # CLI 감지 확인
python cli.py --llm --provider claude_cli --claude-model sonnet
```

앱은 `claude -p --model <모델> --output-format json --tools "" --no-session-persistence` 로 호출합니다.
도구를 모두 끈 상태로 텍스트 분석만 수행하며, 응답의 비용(`total_cost_usd`)을 화면에 표시합니다.

### CLI 옵션

```bash
python cli.py                                  # 통계 엔진 + 감사
python cli.py --llm --provider hf_api --hf-model "Qwen/Qwen3.5-9B:together"
python cli.py --no-optimize --backtest 300 --samples 30000
```

## 2. Streamlit Cloud 배포

1. GitHub 저장소 연결 → Main file: `app.py`
2. Settings → Secrets 에 `STREAMLIT_SECRETS_TEMPLATE.toml` 내용 입력 (HF_TOKEN 은 새로 발급)
3. Claude Code CLI 는 클라우드 컨테이너에 없으므로 사이드바에 "미감지"로 표시됩니다 → **HF Router** 를 사용하세요.

## 3. 산출물 (`outputs/`)

`prediction_result.json`(최종 세트·최적화·필터), `harness_report.json`(감사·AI 리뷰 포함), `backtest_summary.csv`,
`backtest_detail.csv`, `weights.csv`, `number_scores.csv`, `candidates.csv`, `history.csv`

## 4. 테스트

```bash
python -m pytest -q
```

누수 방지(fold 점수 = 과거 구간 재계산), 신호를 심은 합성 데이터에서 최적화기가 신호 전략을 찾는지, Claude CLI 명령 구성,
LLM 제안 클램프 등을 검증합니다.
