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

---

# v7.0 업데이트: 전략 총망라 + 앙상블 최적화 + AI 리뷰 보드 + 모델 선택

## 반영된 v6.4 후보

- [x] LLM JSON 구조화 출력 → 비평가/자문 에이전트 JSON (`extract_json` + 허용범위 클램프)
- [x] LLM 재랭커 대신 **자문 제안 적용 후 재실행** (번호를 LLM 이 직접 조작하지 않음)
- [x] 모델 선택: HF 프리셋/직접입력/fallback, Claude Code CLI(sonnet/opus/haiku/fable/전체 ID)

## 변경 요약

1. 전략 9 → 14개 (markov_transition, multi_scale_freq, last_digit, zone_balance, bonus_carry)
2. 소프트맥스 휴리스틱 가중치 → **로그우도 기반 walk-forward 최적화 + holdout 검증 + 자동 수축**
3. 고정 필터 → 과거 당첨조합 95% 를 포함하는 **학습형 구조 필터** (AC값·구간수·끝수 추가)
4. 단순 overlap 선택 → 커버리지 보너스 greedy
5. StatisticalAuditAgent (LLM 없이 유의성/과적합 판정) 추가
6. LLM 공급자 추상화: `hf_api` | `claude_cli` | `none`
7. VS Code `.vscode/` 실행·디버그·작업 구성

## 다음 v7.1 후보

- 마지막 구간을 완전 봉인한 nested walk-forward (holdout 재사용 편향 제거)
- 자문 제안 → 자동 반복 루프(최대 N회) + 실험 로그 비교표
- 모델별 latency/오류율/비용 로그
