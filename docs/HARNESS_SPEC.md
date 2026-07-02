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
