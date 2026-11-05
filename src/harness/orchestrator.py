from __future__ import annotations

from .base import HarnessContext, AgentSpec
from .agents import (
    PlannerAgent,
    DataIngestionAgent,
    FeatureBacktestAgent,
    VerificationAgent,
    StatisticalAuditAgent,
    AiReviewBoardAgent,
    ReportAgent,
)


def agent_team_spec() -> list[AgentSpec]:
    return [
        AgentSpec("planner", "기획", ["실험 목적 정의", "품질 게이트 설정"], ["config"], ["plan"], ["target_round 확인"]),
        AgentSpec("data_ingestion", "데이터", ["CSV/XLS 파싱", "회차 병합", "유효성 검사"], ["csv", "xls"], ["draws"], ["번호 범위/중복/연속성 검사"]),
        AgentSpec("feature_backtest", "구현/백테스트/최적화", ["14개 전략 점수 계산", "walk-forward 캐싱", "앙상블 가중치·온도 최적화(tune/holdout)", "학습형 구조 필터", "커버리지 기반 세트 선택"], ["draws"], ["pipeline"], ["미래누수 금지", "holdout 검증"]),
        AgentSpec("verification", "검증", ["세트 유효성 검사", "overlap 검사"], ["pipeline"], ["verification"], ["품질 게이트 통과"]),
        AgentSpec("statistical_audit", "통계 감사", ["무작위 대비 유의성(z, Bonferroni)", "과적합 격차", "가중치 집중도", "필터 통과율"], ["pipeline"], ["audit"], ["과장 없는 판정"]),
        AgentSpec("ai_review_board", "AI 리뷰 보드", ["분석가 리포트", "비평가 JSON 판정", "최적화 자문 JSON 제안(허용 범위 클램프)"], ["pipeline", "audit", "LLM provider/model"], ["llm"], ["당첨 보장 표현 금지"]),
        AgentSpec("reporter", "리포트", ["CSV/JSON 저장", "harness report 생성"], ["pipeline", "verification", "audit"], ["outputs"], ["파일 생성 확인"]),
    ]


class AgentHarness:
    def __init__(self, include_llm: bool = False):
        self.agents = [PlannerAgent(), DataIngestionAgent(), FeatureBacktestAgent(), VerificationAgent(), StatisticalAuditAgent()]
        if include_llm:
            self.agents.append(AiReviewBoardAgent())
        self.agents.append(ReportAgent())

    def run(self, ctx: HarnessContext) -> HarnessContext:
        for agent in self.agents:
            ctx.progress(f"▶ {agent.name}")
            result = agent.run(ctx)
            ctx.agent_results.append(result)
            if result.status == "error":
                ctx.log(f"Harness stopped at {agent.name}: {result.message}")
                break
        return ctx
