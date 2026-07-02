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
