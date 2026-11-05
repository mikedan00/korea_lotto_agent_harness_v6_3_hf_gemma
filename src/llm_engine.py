"""LLM 공급자 계층 (v7.0).

- hf_api     : Hugging Face Router (OpenAI 호환 Chat Completions). Streamlit Cloud/로컬 모두 가능.
- claude_cli : 로컬에 설치된 Claude Code CLI(`claude -p`)를 subprocess 로 호출. 로컬 노트북/VS Code 전용.
- none       : LLM 미사용 (통계 엔진 + 결정론적 감사만).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

import requests

from .config import AppConfig

PROVIDERS = {"hf_api": "Hugging Face Router", "claude_cli": "Claude Code CLI (로컬)", "none": "LLM 사용 안 함"}

HF_MODEL_PRESETS = [
    "google/gemma-4-26B-A4B-it:deepinfra",
    "google/gemma-4-26B-A4B-it:novita",
    "google/gemma-4-31B-it:deepinfra",
    "google/gemma-4-31B-it:together",
    "Qwen/Qwen3.5-9B:together",
    "Qwen/Qwen2.5-7B-Instruct:together",
    "meta-llama/Llama-3.3-70B-Instruct:together",
    "deepseek-ai/DeepSeek-V3:together",
]

# Claude Code CLI --model 값: alias(최신 모델로 자동 연결) 또는 전체 모델 ID
CLAUDE_MODEL_PRESETS = [
    "sonnet",
    "opus",
    "haiku",
    "fable",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
]

SYSTEM_GUARD = (
    "You are a rigorous statistical research assistant for Korean Lotto 6/45. "
    "Never claim guaranteed wins or inflated probabilities. Every 6-number combination has the same "
    "1-in-8,145,060 jackpot probability in a fair draw. Answer in Korean."
)


def build_lotto_prompt(result_summary: dict[str, Any]) -> str:
    """v6 호환: 단일 분석 프롬프트."""
    return (
        "당신은 로또 6/45 통계 연구 보조 분석가입니다. "
        "당첨 보장, 확률 과장, 신비주의 표현, 99% 적중 같은 표현을 절대 쓰지 마세요. "
        "아래 JSON의 통계 엔진 결과, 후보 세트, 백테스트 요약, 후보 조합 구조만 근거로 "
        "1) 최종 세트 해석, 2) 약점/리스크, 3) 다음 실험 개선안, 4) 과적합 경고를 한국어로 작성하세요. "
        "번호를 새로 조작하지 말고, 주어진 final_sets를 중심으로 설명하세요.\n\n"
        + json.dumps(result_summary, ensure_ascii=False, indent=2)[:16000]
    )


class HfRouterClient:
    provider = "hf_api"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.url = "https://router.huggingface.co/v1/chat/completions"

    @property
    def model_label(self) -> str:
        return self.cfg.hf_router_model

    def available(self) -> bool:
        return bool(self.cfg.hf_token)

    def _post_chat(self, model: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.hf_temperature,
            "max_tokens": int(max_tokens or self.cfg.hf_max_tokens),
            "stream": False,
        }
        r = requests.post(self.url, headers=headers, json=payload,
                          timeout=(self.cfg.hf_timeout_connect, self.cfg.hf_timeout_read))
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "provider": self.provider, "model": model, "content": content, "raw": data}

    def chat_messages(self, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 이 설정되지 않았습니다."}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = None
        for model in self.cfg.model_candidates:
            for attempt in range(self.cfg.hf_max_retries):
                try:
                    return self._post_chat(model, messages, max_tokens)
                except Exception as e:
                    last_error = f"{model}: {type(e).__name__}: {e}"
                    time.sleep(0.8 * (attempt + 1))
        return {"ok": False, "error": last_error or "LLM 호출 실패"}

    def test_connection(self) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 이 설정되지 않았습니다."}
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "연결 테스트입니다. 한국어로 'HF Router 연결 성공'이라고만 답하세요."},
        ]
        last_error = None
        for model in self.cfg.model_candidates:
            try:
                return self._post_chat(model, messages, max_tokens=80)
            except Exception as e:
                last_error = f"{model}: {type(e).__name__}: {e}"
                time.sleep(0.5)
        return {"ok": False, "error": last_error or "HF 연결 테스트 실패"}

    def chat(self, prompt: str) -> dict[str, Any]:
        return self.chat_messages(SYSTEM_GUARD, prompt)


def detect_claude_cli(path: str = "claude") -> dict[str, Any]:
    """로컬 Claude Code CLI 설치 여부/버전 확인. Streamlit Cloud 에서는 보통 found=False."""
    exe = shutil.which(path)
    if not exe:
        return {"found": False, "path": None, "version": None}
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20,
                             encoding="utf-8", errors="replace")
        version = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None
    except Exception:
        version = None
    return {"found": True, "path": exe, "version": version}


class ClaudeCliClient:
    """`claude -p` 비대화형 모드 호출. 도구는 모두 비활성화(--tools "")하여 텍스트 분석만 수행."""

    provider = "claude_cli"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.exe = shutil.which(cfg.claude_cli_path)

    @property
    def model_label(self) -> str:
        return self.cfg.claude_model

    def available(self) -> bool:
        return self.exe is not None

    def _cmd(self, system: str, minimal: bool = False) -> list[str]:
        cmd = [self.exe, "-p", "--model", self.cfg.claude_model, "--output-format", "json"]
        if not minimal:
            cmd += ["--tools", "", "--no-session-persistence", "--system-prompt", system]
        return cmd

    def _run(self, cmd: list[str], stdin: str) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.cfg.claude_timeout)

    def chat_messages(self, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": f"Claude Code CLI({self.cfg.claude_cli_path})를 찾을 수 없습니다. "
                                          "로컬에서 `npm i -g @anthropic-ai/claude-code` 후 `claude` 로그인 필요."}
        try:
            proc = self._run(self._cmd(system), user)
            if proc.returncode != 0 and "unknown option" in (proc.stderr or "").lower():
                # 구버전 CLI 호환: 최소 옵션으로 재시도 (system 지시는 본문 앞에 포함)
                proc = self._run(self._cmd(system, minimal=True), f"{system}\n\n{user}")
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Claude CLI 타임아웃({self.cfg.claude_timeout}s)"}
        except Exception as e:
            return {"ok": False, "error": f"Claude CLI 실행 오류: {type(e).__name__}: {e}"}
        return parse_claude_json(proc.stdout, proc.stderr, proc.returncode, self.cfg.claude_model)

    def test_connection(self) -> dict[str, Any]:
        return self.chat_messages("You are a concise assistant.", "연결 테스트입니다. 한국어로 'Claude CLI 연결 성공'이라고만 답하세요.")

    def chat(self, prompt: str) -> dict[str, Any]:
        return self.chat_messages(SYSTEM_GUARD, prompt)


def parse_claude_json(stdout: str, stderr: str, returncode: int, model: str) -> dict[str, Any]:
    try:
        data = json.loads(stdout)
    except Exception:
        if returncode == 0 and stdout.strip():
            return {"ok": True, "provider": "claude_cli", "model": model, "content": stdout.strip(), "raw": None}
        return {"ok": False, "error": f"Claude CLI 오류(code={returncode}): {(stderr or stdout)[:1000]}"}
    if data.get("is_error") or returncode != 0:
        return {"ok": False, "error": f"Claude CLI 오류: {str(data.get('result') or stderr)[:1000]}"}
    return {
        "ok": True,
        "provider": "claude_cli",
        "model": model,
        "content": data.get("result", ""),
        "cost_usd": data.get("total_cost_usd"),
        "raw": {k: data.get(k) for k in ("duration_ms", "num_turns", "session_id", "usage")},
    }


def get_llm_client(cfg: AppConfig):
    """cfg.llm_provider 에 맞는 클라이언트. none 이면 None."""
    if cfg.llm_provider == "claude_cli":
        return ClaudeCliClient(cfg)
    if cfg.llm_provider == "hf_api":
        return HfRouterClient(cfg)
    return None


def analyze_with_llm(cfg: AppConfig, result_summary: dict[str, Any]) -> dict[str, Any]:
    """v6 호환: 단일 분석 호출."""
    client = get_llm_client(cfg)
    if client is None:
        return {"ok": False, "error": "LLM 공급자가 none 입니다."}
    return client.chat(build_lotto_prompt(result_summary))


def extract_json(text: str) -> dict[str, Any] | None:
    """LLM 응답에서 첫 JSON 객체를 추출 (```json 코드블록 포함)."""
    if not text:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None
