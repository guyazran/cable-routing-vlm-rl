"""OpenAI-compatible client for the locally served planner (vLLM).

Failure categories are kept distinct (protocol Section 10): a malformed final
answer gets exactly one formatting-only retry; a generation cut off by the token
cap (finish_reason == "length"), a context overflow and a transport error are
reported as such and never retried silently.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..decision import Decision, DecisionFormatError, json_schema, parse_decision


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "Qwen/Qwen3.8-27B"
    base_url: str = "http://127.0.0.1:8000/v1"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 8192
    reasoning_effort: str = "medium"
    enable_thinking: bool = True
    preserve_thinking: bool = False
    structured_output: bool = False  # enable only after the serving smoke check passes
    timeout_s: float = 900.0


@dataclass
class QueryRecord:
    """Everything about one planner decision, for the episode trace."""

    decision: Decision | None = None
    failure: str | None = None  # None | format | cutoff | context_overflow | transport
    attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"decision": self.decision.to_dict() if self.decision else None,
                "failure": self.failure, "attempts": self.attempts}


class PlannerClient:
    def __init__(self, cfg: GenerationConfig = GenerationConfig()):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key="EMPTY", timeout=cfg.timeout_s, max_retries=0)

    def _request(self, messages: list[dict], clip_ids: list[str], seed: int) -> dict[str, Any]:
        c = self.cfg
        extra: dict[str, Any] = {
            "top_k": c.top_k,
            "min_p": c.min_p,
            "chat_template_kwargs": {"enable_thinking": c.enable_thinking,
                                     "preserve_thinking": c.preserve_thinking},
        }
        kwargs: dict[str, Any] = {}
        if c.structured_output:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "decision", "schema": json_schema(clip_ids), "strict": True}}
        t0 = time.monotonic()
        resp = self.client.chat.completions.create(
            model=c.model, messages=messages, temperature=c.temperature, top_p=c.top_p,
            presence_penalty=c.presence_penalty, max_tokens=c.max_tokens, seed=seed,
            reasoning_effort=c.reasoning_effort, extra_body=extra, **kwargs)
        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        return {
            "latency_s": time.monotonic() - t0,
            "finish_reason": choice.finish_reason,
            "content": msg.content or "",
            "reasoning": reasoning,
            "usage": resp.usage.model_dump() if resp.usage else None,
            "model": resp.model,
            "seed": seed,
        }

    def decide(self, messages: list[dict], clip_ids: list[str], seed: int) -> QueryRecord:
        rec = QueryRecord()
        msgs = list(messages)
        for attempt in range(2):
            try:
                r = self._request(msgs, clip_ids, seed + attempt)
            except Exception as e:  # noqa: BLE001 - classified and logged, never swallowed
                text = str(e)
                rec.failure = "context_overflow" if "maximum context length" in text or "too long" in text else "transport"
                rec.attempts.append({"error": f"{type(e).__name__}: {text}"})
                return rec
            rec.attempts.append(r)
            if r["finish_reason"] == "length":
                rec.failure = "cutoff"
                return rec
            try:
                rec.decision = parse_decision(r["content"], clip_ids)
                rec.failure = None
                return rec
            except DecisionFormatError as e:
                r["format_error"] = str(e)
                rec.failure = "format"
                if attempt == 0:
                    msgs = msgs + [
                        {"role": "assistant", "content": r["content"]},
                        {"role": "user", "content": (
                            "Your final answer could not be parsed: " + str(e) + ". Reply with only one JSON "
                            "object matching this schema, and nothing else:\n" + json.dumps(json_schema(clip_ids)))},
                    ]
        return rec
