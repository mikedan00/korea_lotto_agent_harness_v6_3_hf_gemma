from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness
from src.llm_engine import CLAUDE_MODEL_PRESETS, HF_MODEL_PRESETS, PROVIDERS, detect_claude_cli


def main():
    ap = argparse.ArgumentParser(description="Korea Lotto 6/45 Agent Harness v7.0")
    ap.add_argument("--csv", default="data/korealotto_1224V_CSV2.csv")
    ap.add_argument("--xls", default="data/LottoSpecial_공나온순서_1225.xls")
    ap.add_argument("--target", type=int, default=1226)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--backtest", type=int, default=180)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-optimize", action="store_true", help="앙상블 최적화 끄기(균등 가중치)")
    ap.add_argument("--llm", action="store_true", help="AI 리뷰 보드 실행 (--provider 로 공급자 선택)")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None, help="hf_api | claude_cli | none")
    ap.add_argument("--hf-model", default=None, help="HF Router 모델 (예: google/gemma-4-26B-A4B-it:deepinfra)")
    ap.add_argument("--claude-model", default=None, help="Claude CLI 모델 (sonnet/opus/haiku 또는 전체 ID)")
    ap.add_argument("--list-models", action="store_true", help="공급자/모델 프리셋과 Claude CLI 감지 결과 출력")
    args = ap.parse_args()

    if args.list_models:
        print("[Providers]", json.dumps(PROVIDERS, ensure_ascii=False))
        print("[HF presets]\n  " + "\n  ".join(HF_MODEL_PRESETS))
        print("[Claude CLI presets]\n  " + "\n  ".join(CLAUDE_MODEL_PRESETS))
        print("[Claude CLI detect]", detect_claude_cli())
        return

    cfg = AppConfig.from_env(
        target_round=args.target,
        n_sets=args.sets,
        candidate_samples=args.samples,
        backtest_window=args.backtest,
        output_dir=args.outdir,
        optimize=not args.no_optimize,
        llm_provider=args.provider,
        hf_router_model=args.hf_model,
        claude_model=args.claude_model,
    )
    ctx = HarnessContext(cfg=cfg, csv_path=args.csv, xls_path=args.xls, on_progress=lambda m: print(f"  · {m}"))
    ctx = AgentHarness(include_llm=args.llm).run(ctx)

    result = ctx.artifacts.get("pipeline", {})
    opt = result.get("optimization", {})
    print(f"\n[{cfg.target_round}회 후보 {cfg.n_sets}세트]")
    for i, s in enumerate(result.get("final_sets", []), 1):
        print(f"Set {i}: {', '.join(map(str, s))}")
    if opt:
        print(f"\n[최적화] holdout edge={opt['holdout_edge_nats']:+.4f} nats | holdout 평균적중={opt['holdout_mean_hits']:.3f}"
              f" (무작위 0.800) | 수축={opt['shrunk_to_uniform']}")
    audit = ctx.artifacts.get("audit")
    if audit:
        print(f"[통계 감사] verdict={audit['verdict']}")
        for f in audit["risk_flags"]:
            print(f"  - {f}")
    llm = ctx.artifacts.get("llm")
    if llm and llm.get("ok"):
        print(f"\n[AI 리뷰: {llm['provider']} / {llm['model']}]")
        print((llm["steps"].get("analyst", {}).get("content") or "")[:3000])
        if llm.get("critic"):
            print("\n[비평가]", json.dumps(llm["critic"], ensure_ascii=False, indent=1))
        if llm.get("suggestions"):
            print("[자문 제안(클램프 적용)]", json.dumps(llm["suggestions"], ensure_ascii=False))
    print("\n[Agent Results]")
    for ar in ctx.agent_results:
        print(f"- {ar.name}: {ar.status} | {ar.message}")
    print(f"\noutputs: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()
