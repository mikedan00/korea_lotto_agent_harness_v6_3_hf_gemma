# v7.0 알고리즘 명세

## 1. 전략 (src/strategies.py)

모든 전략은 `score(draws) -> 45개 번호 z-score` 이며, 입력 `draws` 이외의 정보를 쓰지 않습니다.

| 전략 | 신호 |
|---|---|
| long_frequency | 전체 출현 빈도 |
| recent_decay | 반감기 45회 지수감쇠 빈도 |
| gap | 현재 미출현 간격 / 평균 간격 |
| hot_cold | 최근 30회 75% + 장기 25% |
| momentum | 최근 30회 출현률 − 전체 출현률 |
| bayes | Beta-Binomial 사후평균 |
| pair_affinity | 평균 pair 동시출현 (수축) |
| prime_factor | 소인수 성분의 장기/최근/갭 신호 |
| anti_crowd | 직전 반복 회피 + 생일수 편향 보정 |
| **markov_transition** | 직전 회차 번호 m → 다음 회차 n 전이확률 lift (Laplace) |
| **multi_scale_freq** | 10/25/50/100회 Dirichlet 평활 빈도 평균 |
| **last_digit** | 끝수 최근 50회 출현 lift |
| **zone_balance** | 번호대 최근 30회 과소출현 평균회귀 |
| **bonus_carry** | 최근 보너스 번호 감쇠 신호 |

참고: `pair_affinity` 는 정의상 `long_frequency` 와 거의 같은 순위를 냅니다(번호별 pair 합 = 5×출현수). 최적화기가 중복 가중치를 알아서 조정합니다.

## 2. Walk-forward 캐싱과 최적화 (src/optimizer.py)

- fold t: `draws[:t]` 로 14개 전략 점수 → 텐서 `S[F,K,45]`, 정답 `Y[F,45]`
- 앙상블: `q = softmax(β · Σ_k w_k S_k)`, w 는 단체(simplex) 위
- 목적함수: `mean_t Σ_{n∈Y_t} log q_n − 6·log(1/45)` (nats, 균등 대비 edge) − `λ‖w − 1/K‖²`
- 탐색: 균등/원-핫/Dirichlet(0.6) 300개 × β 격자 → 가우시안 섭동 hill-climb 200회
- 검증: 앞 70% fold 로 탐색, 뒤 30% fold(holdout) 로 평가. holdout edge ≤ 0 이면 β×0.25 수축 + 후보 점수의 확률 성분 0.5배
- 유의성: 전략별 top-6 평균적중 vs 0.8, 분산 0.6145/F 의 z-검정(단측), Bonferroni ×K

한계(AI 비평가 지적 반영): holdout 은 수축 여부 결정에도 쓰이므로 완전한 외표본은 아닙니다.
더 엄격한 검증이 필요하면 `backtest_window` 를 늘리고 마지막 구간을 별도로 남겨두는 실험을 권장합니다.

## 3. 후보 생성 (src/engine.py)

1. 샘플링 분포 `softmax(τ · ensemble_z)` 에서 Gumbel top-k 로 `samples×4` 조합 생성 → 중복 제거
2. 학습형 필터: 과거 당첨조합 지표의 2.5%~97.5% 구간 (coverage 조절 가능), 결합 통과율을 진단값으로 기록
3. 점수 = `prob·z(Σlog q) + pair·z(Σ log pair lift) + typical·z(−mean|z_metric|) + prime·z(소인수 전형성)`
4. 세트 선택: 상위 400개 중 `score + coverage_weight × 신규번호수` greedy, overlap ≤ max_overlap

## 4. 리뷰 계층 (src/harness/review.py)

- `statistical_audit`: verdict ∈ {signal, weak_signal, no_evidence_of_edge} + risk_flags
- AI 리뷰 보드: analyst → critic(JSON) → advisor(JSON). advisor 제안은 `TUNABLE` 허용 키/범위로 클램프 후에만 적용 가능
