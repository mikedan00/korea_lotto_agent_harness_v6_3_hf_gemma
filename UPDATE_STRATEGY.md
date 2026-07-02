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
