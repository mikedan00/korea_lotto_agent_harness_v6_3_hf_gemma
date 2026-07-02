from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import AppConfig


def build_lotto_prompt(result_summary: dict[str, Any]) -> str:
    return (
        "당신은 로또 6/45 통계 연구 보조 분석가입니다. "
        "당첨 보장, 확률 과장, 신비주의 표현, 99% 적중 같은 표현을 절대 쓰지 마세요. "
        "아래 JSON의 통계 엔진 결과, 후보 세트, 백테스트 요약, 후보 조합 구조만 근거로 "
        "1) 최종 세트 해석, 2) 약점/리스크, 3) 다음 실험 개선안, 4) 과적합 경고를 한국어로 작성하세요. "
        "번호를 새로 조작하지 말고, 주어진 final_sets를 중심으로 설명하세요.\n\n"
        + json.dumps(result_summary, ensure_ascii=False, indent=2)[:16000]
    )


class HfRouterClient:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.url = "https://router.huggingface.co/v1/chat/completions"

    def available(self) -> bool:
        return self.cfg.llm_engine == "hf_api" and bool(self.cfg.hf_token)

    def _post_chat(self, model: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.hf_temperature,
            "max_tokens": int(max_tokens or self.cfg.hf_max_tokens),
            "stream": False,
        }
        r = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=(self.cfg.hf_timeout_connect, self.cfg.hf_timeout_read),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "model": model, "content": content, "raw": data}

    def test_connection(self) -> dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 또는 LLM_ENGINE 설정이 없습니다."}
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
        if not self.available():
            return {"ok": False, "error": "HF_TOKEN 또는 LLM_ENGINE 설정이 없습니다."}
        messages = [
            {"role": "system", "content": "You are a rigorous statistical research assistant. Avoid any guaranteed lottery claims."},
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for model in self.cfg.model_candidates:
            for attempt in range(self.cfg.hf_max_retries):
                try:
                    return self._post_chat(model, messages)
                except Exception as e:
                    last_error = f"{model}: {type(e).__name__}: {e}"
                    time.sleep(0.8 * (attempt + 1))
        return {"ok": False, "error": last_error or "LLM 호출 실패"}


def analyze_with_llm(cfg: AppConfig, result_summary: dict[str, Any]) -> dict[str, Any]:
    client = HfRouterClient(cfg)
    return client.chat(build_lotto_prompt(result_summary))
