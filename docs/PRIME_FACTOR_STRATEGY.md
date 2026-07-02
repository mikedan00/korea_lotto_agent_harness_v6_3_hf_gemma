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
