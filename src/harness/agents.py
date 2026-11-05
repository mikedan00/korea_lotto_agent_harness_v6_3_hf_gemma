from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import AgentResult, HarnessContext
from ..data_loader import load_history, validate_draws
from ..engine import run_core_pipeline, save_pipeline_outputs
from ..llm_engine import get_llm_client
from .review import run_review_board, statistical_audit


class PlannerAgent:
    name = "planner"
    role = "기획/실험 설계 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        r = AgentResult(self.name, "ok", "실험 계획 수립 완료")
        ctx.artifacts["plan"] = {
            "target_round": ctx.cfg.target_round,
            "n_sets": ctx.cfg.n_sets,
            "backtest_window": ctx.cfg.backtest_window,
            "candidate_samples": ctx.cfg.candidate_samples,
            "optimize": ctx.cfg.optimize,
            "llm_provider": ctx.cfg.llm_provider,
            "quality_gates": [
                "데이터 유효성 오류 0건",
                "holdout 구간으로 앙상블 과적합 검증",
                "미래 데이터 누수 금지",
                "최종 세트 개수 일치",
                "세트 간 overlap 제한 충족",
                "당첨 보장 표현 금지",
            ],
        }
        ctx.log("PlannerAgent: plan created")
        return r.finish()


class DataIngestionAgent:
    name = "data_ingestion"
    role = "데이터 수집/정규화 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        draws = load_history(ctx.csv_path, ctx.xls_path)
        ctx.data["draws"] = draws
        msg = f"{draws[0].round}~{draws[-1].round}회, 총 {len(draws)}개 회차 로드"
        ctx.log(f"DataIngestionAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"round_min": draws[0].round, "round_max": draws[-1].round, "rows": len(draws)}).finish()


class FeatureBacktestAgent:
    name = "feature_backtest"
    role = "전략 구현/백테스트 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        draws = ctx.data["draws"]
        result = run_core_pipeline(draws, ctx.cfg, progress=ctx.progress)
        ctx.artifacts["pipeline"] = result
        summary = result["backtest_summary"]
        opt = result["optimization"]
        msg = (f"전략 {len(summary)}개 walk-forward 백테스트 + 앙상블 최적화 완료 "
               f"(holdout edge {opt['holdout'].get('edge_nats', opt['holdout_edge_nats']):+.4f} nats, 수축={opt['shrunk_to_uniform']})")
        ctx.log(f"FeatureBacktestAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"strategy_count": len(summary), "candidate_count": len(result["candidates"])}).finish()


class VerificationAgent:
    name = "verification"
    role = "검증/품질 게이트 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts["pipeline"]
        final_sets = result["final_sets"]
        errors = []
        if len(final_sets) != ctx.cfg.n_sets:
            errors.append(f"최종 세트 개수 불일치: {len(final_sets)}")
        for s in final_sets:
            if len(s) != 6 or len(set(s)) != 6 or not all(1 <= n <= 45 for n in s):
                errors.append(f"번호 유효성 오류: {s}")
        for i in range(len(final_sets)):
            for j in range(i + 1, len(final_sets)):
                if len(set(final_sets[i]) & set(final_sets[j])) > ctx.cfg.max_overlap:
                    errors.append(f"overlap 초과: {final_sets[i]} vs {final_sets[j]}")
        status = "ok" if not errors else "error"
        msg = "품질 게이트 통과" if not errors else "; ".join(errors)
        ctx.artifacts["verification"] = {"status": status, "errors": errors}
        ctx.log(f"VerificationAgent: {msg}")
        return AgentResult(self.name, status, msg, {"errors": errors}).finish()


class StatisticalAuditAgent:
    name = "statistical_audit"
    role = "결정론적 통계 감사 에이전트 (LLM 불필요)"
    def run(self, ctx: HarnessContext) -> AgentResult:
        audit = statistical_audit(ctx.artifacts["pipeline"], ctx.cfg)
        ctx.artifacts["audit"] = audit
        msg = f"판정={audit['verdict']}, 리스크 {len(audit['risk_flags'])}건"
        ctx.log(f"StatisticalAuditAgent: {msg}")
        return AgentResult(self.name, "ok", msg, {"verdict": audit["verdict"]}).finish()


class AiReviewBoardAgent:
    """선택한 LLM(HF Router 또는 Claude Code CLI)으로 분석가 → 비평가 → 최적화 자문 3단계 리뷰."""
    name = "ai_review_board"
    role = "AI 리뷰 보드 (분석/비평/최적화 자문)"
    def run(self, ctx: HarnessContext) -> AgentResult:
        client = get_llm_client(ctx.cfg)
        if client is None:
            ctx.artifacts["llm"] = {"ok": False, "error": "LLM 공급자 none"}
            return AgentResult(self.name, "skipped", "LLM 공급자 none: AI 리뷰 생략").finish()
        if not client.available():
            msg = ("HF_TOKEN 미설정" if client.provider == "hf_api" else "Claude Code CLI 미설치/PATH 없음") + ": AI 리뷰 생략"
            ctx.artifacts["llm"] = {"ok": False, "error": msg}
            return AgentResult(self.name, "skipped", msg).finish()
        board = run_review_board(client, ctx.artifacts["pipeline"], ctx.cfg, ctx.artifacts.get("audit", {}), progress=ctx.progress)
        ctx.artifacts["llm"] = board
        if not board.get("ok"):
            return AgentResult(self.name, "warning", str(board.get("error"))).finish()
        verdict = (board.get("critic") or {}).get("verdict", "n/a")
        msg = f"{board['provider']}:{board['model']} 리뷰 완료 (비평 verdict={verdict}, 제안 {len(board.get('suggestions', {}))}건)"
        ctx.log(f"AiReviewBoardAgent: {msg}")
        return AgentResult(self.name, "ok", msg).finish()


# v6 호환 이름
LlmAnalystAgent = AiReviewBoardAgent


class ReportAgent:
    name = "reporter"
    role = "리포트/산출물 저장 에이전트"
    def run(self, ctx: HarnessContext) -> AgentResult:
        result = ctx.artifacts["pipeline"]
        paths = save_pipeline_outputs(result, ctx.cfg.output_dir)
        report = {
            "plan": ctx.artifacts.get("plan"),
            "final_sets": result.get("final_sets"),
            "probability": result.get("probability"),
            "verification": ctx.artifacts.get("verification"),
            "optimization": result.get("optimization"),
            "audit": ctx.artifacts.get("audit"),
            "agent_results": [a.__dict__ for a in ctx.agent_results],
            "llm": ctx.artifacts.get("llm"),
        }
        p = Path(ctx.cfg.output_dir) / "harness_report.json"
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        paths["harness_report"] = str(p)
        ctx.artifacts["paths"] = paths
        ctx.log("ReportAgent: artifacts saved")
        return AgentResult(self.name, "ok", "리포트 및 CSV/JSON 산출물 저장 완료", paths).finish()
