from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _streamlit_secret(key: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        return None
    return None


def get_setting(key: str, default: str = "") -> str:
    val = _streamlit_secret(key)
    if val is not None and val != "":
        return val
    return os.getenv(key, default)


@dataclass
class AppConfig:
    target_round: int = 1226
    n_sets: int = 2
    candidate_samples: int = 12000
    backtest_window: int = 180
    min_train_rounds: int = 200
    seed: int = 20260530
    max_overlap: int = 2
    sum_min: int = 98
    sum_max: int = 177
    output_dir: str = "outputs"

    # v7.0 최적화/후보 생성
    optimize: bool = True
    tune_ratio: float = 0.7
    opt_random: int = 300
    opt_refine: int = 200
    use_learned_filters: bool = True
    filter_coverage: float = 0.95
    explore_temperature: float = 0.35
    coverage_weight: float = 0.3
    score_weights: dict = field(default_factory=lambda: {"prob": 1.0, "pair": 0.25, "typical": 0.35, "prime": 0.15})

    # LLM 공급자: hf_api | claude_cli | none
    llm_provider: str = "hf_api"
    claude_model: str = "sonnet"
    claude_cli_path: str = "claude"
    claude_timeout: int = 240

    llm_engine: str = "hf_api"
    hf_token: str = ""
    hf_router_model: str = "google/gemma-4-26B-A4B-it:deepinfra"
    hf_model_candidates: str = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
    hf_max_tokens: int = 1400
    hf_temperature: float = 0.2
    hf_timeout_connect: int = 10
    hf_timeout_read: int = 120
    hf_max_retries: int = 3

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppConfig":
        cfg = cls(
            llm_provider=get_setting("LLM_PROVIDER", get_setting("LLM_ENGINE", "hf_api")),
            claude_model=get_setting("CLAUDE_MODEL", "sonnet"),
            claude_cli_path=get_setting("CLAUDE_CLI_PATH", "claude"),
            claude_timeout=int(float(get_setting("CLAUDE_TIMEOUT", "240"))),
            llm_engine=get_setting("LLM_ENGINE", "hf_api"),
            hf_token=get_setting("HF_TOKEN", ""),
            hf_router_model=get_setting("HF_ROUTER_MODEL", get_setting("HF_MODEL_ID", "google/gemma-4-26B-A4B-it:deepinfra")),
            hf_model_candidates=get_setting("HF_MODEL_CANDIDATES", cls.hf_model_candidates),
            hf_max_tokens=int(float(get_setting("HF_MAX_TOKENS", "1400"))),
            hf_temperature=float(get_setting("HF_TEMPERATURE", "0.2")),
            hf_timeout_connect=int(float(get_setting("HF_TIMEOUT_CONNECT", "10"))),
            hf_timeout_read=int(float(get_setting("HF_TIMEOUT_READ", "120"))),
            hf_max_retries=int(float(get_setting("HF_MAX_RETRIES", "3"))),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    @property
    def model_candidates(self) -> list[str]:
        seen = []
        for m in [self.hf_router_model] + [x.strip() for x in self.hf_model_candidates.split(",") if x.strip()]:
            if m not in seen:
                seen.append(m)
        return seen
