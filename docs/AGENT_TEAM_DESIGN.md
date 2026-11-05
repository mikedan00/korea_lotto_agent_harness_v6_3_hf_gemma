# 에이전트 팀 설계

이 프로젝트는 단일 함수형 예측기가 아니라 **Agent Harness**입니다. 각 에이전트는 입력, 출력, 품질 게이트를 갖고 순차적으로 실행됩니다.

## 팀 구성

| 에이전트 | 역할 | 핵심 산출물 |
|---|---|---|
| PlannerAgent | 연구 계획, 품질 게이트 정의 | plan |
| DataIngestionAgent | CSV/XLS 로딩, 정규화, 검증 | draws |
| FeatureBacktestAgent | 14개 전략, walk-forward 캐싱, 앙상블 최적화(tune/holdout), 학습형 필터, 후보 생성 | pipeline |
| VerificationAgent | 최종 세트 유효성, 중복도 검증 | verification |
| StatisticalAuditAgent | 무작위 대비 유의성, 과적합 격차, 가중치 집중, 필터 통과율 (LLM 불필요) | audit |
| AiReviewBoardAgent | HF Router 또는 Claude Code CLI 로 분석가 → 비평가(JSON) → 최적화 자문(JSON) | llm |
| ReportAgent | CSV/JSON 산출물 저장 | outputs |

## 하네스 철학

- 번호 추천보다 **검증 가능한 실험 루틴**이 우선입니다.
- rolling-forward 외의 무작위 K-fold는 기본 금지합니다.
- LLM은 예측 엔진이 아니라 리포트/비판/설명 계층입니다.
- 당첨 보장 표현은 모든 산출물에서 금지합니다.
