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
