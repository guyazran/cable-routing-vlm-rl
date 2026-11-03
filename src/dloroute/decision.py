"""Structured high-level decision (protocol Section 6).

A decision is either a skill command on a visible clip or a completion flag.
Validation is purely syntactic: a well-formed but wrong target, skill, or
completion flag is passed through unchanged so it is scored as a decision
error, never silently corrected.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

SKILLS = ("Insert", "Pull", "Flatten")


@dataclass(frozen=True)
class Decision:
    skill: str | None
    target_clip: str | None
    done: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionFormatError(ValueError):
    """Final response is not a valid decision; message is fed back on the one retry."""


def json_schema(clip_ids: list[str]) -> dict[str, Any]:
    """JSON schema for schema-constrained decoding (enum over visible clips)."""
    return {
        "type": "object",
        "properties": {
            "skill": {"anyOf": [{"type": "string", "enum": list(SKILLS)}, {"type": "null"}]},
            "target_clip": {"anyOf": [{"type": "string", "enum": list(clip_ids)}, {"type": "null"}]},
            "done": {"type": "boolean"},
        },
        "required": ["skill", "target_clip", "done"],
        "additionalProperties": False,
    }


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    if not candidates:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise DecisionFormatError("no JSON object found in the final response")
        candidates = [text[start : end + 1]]
    if len(candidates) > 1:
        raise DecisionFormatError("multiple JSON objects found; return exactly one decision")
    try:
        obj = json.loads(candidates[0])
    except json.JSONDecodeError as e:
        raise DecisionFormatError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise DecisionFormatError("decision must be a JSON object")
    return obj


def parse_decision(text: str, clip_ids: list[str]) -> Decision:
    """Parse and syntactically validate a final (non-reasoning) model response."""
    obj = _extract_json_object(text)
    missing = {"skill", "target_clip", "done"} - obj.keys()
    extra = obj.keys() - {"skill", "target_clip", "done"}
    if missing:
        raise DecisionFormatError(f"missing keys: {sorted(missing)}")
    if extra:
        raise DecisionFormatError(f"unexpected keys: {sorted(extra)}")
    done, skill, target = obj["done"], obj["skill"], obj["target_clip"]
    if not isinstance(done, bool):
        raise DecisionFormatError("'done' must be true or false")
    if done:
        if skill is not None or target is not None:
            raise DecisionFormatError("when 'done' is true, 'skill' and 'target_clip' must be null")
        return Decision(None, None, True)
    if skill not in SKILLS:
        raise DecisionFormatError(f"'skill' must be one of {list(SKILLS)} when 'done' is false")
    if target not in clip_ids:
        raise DecisionFormatError(f"'target_clip' must be one of {clip_ids} when 'done' is false")
    return Decision(skill, target, False)
