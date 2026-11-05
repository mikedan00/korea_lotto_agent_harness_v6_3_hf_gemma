import json
import subprocess

import numpy as np

from src.config import AppConfig
from src.data_loader import Draw, load_history
from src.engine import combo_metrics_matrix, learn_filters, run_core_pipeline, select_final_sets
from src.harness.review import apply_suggestions, sanitize_suggestions, statistical_audit
from src.llm_engine import ClaudeCliClient, extract_json, get_llm_client, parse_claude_json
from src.optimizer import FoldCache, mean_edge, optimize_ensemble
from src.strategies import ac_value, default_strategies

DRAWS = load_history("data/korealotto_1224V_CSV2.csv")


def test_all_strategies_return_finite_45_scores():
    for s in default_strategies():
        out = s.score(DRAWS[:300])
        assert set(out.scores) == set(range(1, 46)), s.name
        assert np.isfinite(list(out.scores.values())).all(), s.name


def test_fold_cache_has_no_future_leakage():
    # fold i 의 점수는 draws[:start+i] 만으로 계산한 점수와 같아야 하고, 정답은 draws[start+i] 여야 한다
    from src.optimizer import collect_fold_scores
    strategies = default_strategies()
    start = len(DRAWS) - 3
    cache = collect_fold_scores(DRAWS, strategies, start)
    for i in range(3):
        expected = [[s.score(DRAWS[: start + i]).scores[n] for n in range(1, 46)] for s in strategies]
        assert np.allclose(cache.scores[i], expected)
        assert set(np.flatnonzero(cache.actual[i]) + 1) == set(DRAWS[start + i].numbers)


def test_optimizer_finds_planted_signal():
    rng = np.random.default_rng(0)
    f, k = 200, 3
    actual = np.zeros((f, 45))
    scores = rng.normal(size=(f, k, 45))
    for i in range(f):
        nums = rng.choice(45, 6, replace=False)
        actual[i, nums] = 1
        scores[i, 1, nums] += 1.5  # 전략 1만 실제 신호 보유
    cache = FoldCache(list(range(f)), ["noise_a", "signal", "noise_b"], scores, actual)
    fit = optimize_ensemble(cache, n_random=100, n_refine=100)
    assert fit.holdout_edge > 0
    assert not fit.shrunk
    assert max(fit.weights, key=fit.weights.get) == "signal"


def test_uniform_beta_zero_edge_is_zero():
    cache_scores = np.random.default_rng(1).normal(size=(10, 2, 45))
    actual = np.zeros((10, 45)); actual[:, :6] = 1
    assert abs(mean_edge(cache_scores, actual, np.array([0.5, 0.5]), 0.0)) < 1e-9


def test_combo_metrics_and_ac():
    m = combo_metrics_matrix(np.array([[1, 2, 3, 10, 20, 45]]), (1, 5, 9, 11, 13, 15))
    assert m["sum"][0] == 81 and m["consecutive_pairs"][0] == 2 and m["last_overlap"][0] == 1
    assert m["ac"][0] == ac_value([1, 2, 3, 10, 20, 45])
    f = learn_filters(DRAWS)
    assert f["_joint_pass_rate"]["value"] > 0.7


def test_select_final_sets_respects_overlap():
    import pandas as pd
    df = pd.DataFrame({"numbers": ["1, 2, 3, 4, 5, 6", "1, 2, 3, 4, 5, 7", "10, 11, 12, 13, 14, 15"], "score": [3, 2.9, 1]})
    sets = select_final_sets(df, 2, max_overlap=2)
    assert sets == [[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]


def test_pipeline_smoke_and_audit():
    cfg = AppConfig.from_env(backtest_window=40, candidate_samples=2000, opt_random=30, opt_refine=30, output_dir="outputs")
    r = run_core_pipeline(DRAWS, cfg)
    assert len(r["final_sets"]) == cfg.n_sets
    for s in r["final_sets"]:
        assert len(set(s)) == 6 and all(1 <= n <= 45 for n in s)
    assert abs(sum(r["optimization"]["weights"].values()) - 1) < 1e-6
    audit = statistical_audit(r, cfg)
    assert audit["verdict"] in {"signal", "weak_signal", "no_evidence_of_edge"}


def test_extract_json_and_sanitize():
    txt = 'blah ```json\n{"param_suggestions": {"backtest_window": 5000, "evil": 1, "tune_ratio": "0.6"}, "score_weights": {"prob": 9, "x": 1}}\n```'
    adv = extract_json(txt)
    s = sanitize_suggestions(adv)
    assert s == {"backtest_window": 600, "tune_ratio": 0.6, "score_weights": {"prob": 2.0}}
    cfg = apply_suggestions(AppConfig(), s)
    assert cfg.backtest_window == 600 and cfg.score_weights["prob"] == 2.0 and cfg.score_weights["pair"] == 0.25


def test_parse_claude_json():
    ok = parse_claude_json(json.dumps({"type": "result", "result": "안녕", "is_error": False, "total_cost_usd": 0.01}), "", 0, "sonnet")
    assert ok["ok"] and ok["content"] == "안녕" and ok["cost_usd"] == 0.01
    bad = parse_claude_json(json.dumps({"result": "auth error", "is_error": True}), "", 1, "sonnet")
    assert not bad["ok"]


def test_claude_cli_client_builds_command(monkeypatch):
    seen = {}

    def fake_run(cmd, input, **kw):
        seen["cmd"], seen["input"] = cmd, input
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": "ok", "is_error": False}), "")

    cfg = AppConfig(llm_provider="claude_cli", claude_model="opus")
    client = get_llm_client(cfg)
    assert isinstance(client, ClaudeCliClient)
    client.exe = "/usr/bin/claude"
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = client.chat_messages("SYS", "USER")
    assert r["ok"] and r["content"] == "ok"
    assert seen["cmd"][:4] == ["/usr/bin/claude", "-p", "--model", "opus"]
    assert "--tools" in seen["cmd"] and seen["input"] == "USER"


def test_none_provider():
    assert get_llm_client(AppConfig(llm_provider="none")) is None
