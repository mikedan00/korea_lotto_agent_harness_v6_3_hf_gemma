from __future__ import annotations

from .base import HarnessContext, AgentSpec
from .agents import PlannerAgent, DataIngestionAgent, FeatureBacktestAgent, VerificationAgent, LlmAnalystAgent, ReportAgent


def agent_team_spec() -> list[AgentSpec]:
    return [
        AgentSpec("planner", "기획", ["실험 목적 정의", "품질 게이트 설정"], ["config"], ["plan"], ["target_round 확인"]),
        AgentSpec("data_ingestion", "데이터", ["CSV/XLS 파싱", "회차 병합", "유효성 검사"], ["csv", "xls"], ["draws"], ["번호 범위/중복/연속성 검사"]),
        AgentSpec("feature_backtest", "구현/백테스트", ["전략 점수 계산", "rolling-forward 백테스트", "후보 조합 생성"], ["draws"], ["pipeline"], ["미래누수 금지"]),
        AgentSpec("verification", "검증", ["세트 유효성 검사", "overlap 검사", "확률 요약 확인"], ["pipeline"], ["verification"], ["품질 게이트 통과"]),
        AgentSpec("llm_analyst", "AI 분석", ["LLM 보조 해석", "과장 표현 제거"], ["pipeline", "HF settings"], ["llm"], ["당첨 보장 표현 금지"]),
        AgentSpec("reporter", "리포트", ["CSV/JSON 저장", "harness report 생성"], ["pipeline", "verification"], ["outputs"], ["파일 생성 확인"]),
    ]


class AgentHarness:
    def __init__(self, include_llm: bool = False):
        self.agents = [PlannerAgent(), DataIngestionAgent(), FeatureBacktestAgent(), VerificationAgent()]
        if include_llm:
            self.agents.append(LlmAnalystAgent())
        self.agents.append(ReportAgent())

    def run(self, ctx: HarnessContext) -> HarnessContext:
        for agent in self.agents:
            result = agent.run(ctx)
            ctx.agent_results.append(result)
            if result.status == "error":
                ctx.log(f"Harness stopped at {agent.name}: {result.message}")
                break
        return ctx
