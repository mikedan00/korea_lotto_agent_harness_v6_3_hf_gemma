from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness


def main():
    ap = argparse.ArgumentParser(description="Korea Lotto 6/45 Agent Harness v6")
    ap.add_argument("--csv", default="data/korealotto_1224V_CSV2.csv")
    ap.add_argument("--xls", default="data/LottoSpecial_공나온순서_1225.xls")
    ap.add_argument("--target", type=int, default=1226)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--backtest", type=int, default=180)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig.from_env(
        target_round=args.target,
        n_sets=args.sets,
        candidate_samples=args.samples,
        backtest_window=args.backtest,
        output_dir=args.outdir,
    )
    ctx = HarnessContext(cfg=cfg, csv_path=args.csv, xls_path=args.xls)
    ctx = AgentHarness(include_llm=args.llm).run(ctx)

    result = ctx.artifacts.get("pipeline", {})
    print(f"\n[1226 후보 {cfg.n_sets}세트]")
    for i, s in enumerate(result.get("final_sets", []), 1):
        print(f"Set {i}: {', '.join(map(str, s))}")
    print("\n[Agent Results]")
    for ar in ctx.agent_results:
        print(f"- {ar.name}: {ar.status} | {ar.message}")
    print(f"\noutputs: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()
