from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class AgentResult:
    name: str
    status: str
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: str = ""

    def finish(self) -> "AgentResult":
        self.ended_at = datetime.utcnow().isoformat()
        return self


@dataclass
class HarnessContext:
    cfg: Any
    csv_path: str | None = None
    xls_path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.logs.append(f"{datetime.utcnow().isoformat()} | {msg}")


class HarnessAgent(Protocol):
    name: str
    role: str
    def run(self, ctx: HarnessContext) -> AgentResult:
        ...


@dataclass
class AgentSpec:
    name: str
    role: str
    responsibilities: list[str]
    inputs: list[str]
    outputs: list[str]
    quality_gates: list[str]
