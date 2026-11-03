"""Planner serving checks before integration (protocol Section 4, items 1-5).

Run against a live server (scripts/serve_vlm.sh main):

    conda activate cable
    python scripts/smoke/vlm_smoke.py --scene runs/smoke/isaac/full.png --crop runs/smoke/isaac/crop_C01.png \
        --out runs/smoke/vlm

This is a serving/infrastructure check, not a perception evaluation: the
development image set for state recognition (Section 4) is separate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cablerouting.config import load_config
from dloroute.decision import parse_decision
from dloroute.planner.client import GenerationConfig, PlannerClient
from dloroute.planner.prompts import HistoryEntry, PlannerState, PromptBuilder, PromptConfig, png_data_url


def gpu_memory() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def glyph(text: str, color, size=(640, 480)) -> Image.Image:
    im = Image.new("RGB", size, (128, 128, 128))
    d = ImageDraw.Draw(im)
    d.text((size[0] // 2, size[1] // 2), text, fill=color, font=ImageFont.load_default(size=200), anchor="mm")
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--examples", nargs=2, help="two example images (default: reuse the crop)")
    ap.add_argument("--out", default="runs/smoke/vlm")
    ap.add_argument("--structured", action="store_true", help="also test schema-constrained decoding")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    gen: GenerationConfig = cfg.generation
    client = PlannerClient(gen)
    report: dict = {"generation": gen.__dict__}

    # 1. Model and revision actually served.
    models = client.client.models.list()
    report["served_models"] = [m.model_dump() for m in models.data]
    report["gpu_memory_mib_loaded"] = gpu_memory()

    # 2. Both images influence the answer.
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Image A:"}, {"type": "image_url", "image_url": {"url": png_data_url(glyph("7", (200, 30, 30)))}},
        {"type": "text", "text": "Image B:"}, {"type": "image_url", "image_url": {"url": png_data_url(glyph("K", (30, 30, 200)))}},
        {"type": "text", "text": "What character is shown in image A, and what character in image B? Also name each "
                                 "character's colour. Answer as JSON {\"A\": ..., \"B\": ..., \"A_color\": ..., \"B_color\": ...}."}]}]
    r = client._request(msgs, [], seed=1)
    report["two_image_check"] = {"content": r["content"], "finish": r["finish_reason"], "usage": r["usage"],
                                 "latency_s": r["latency_s"],
                                 "passed": "7" in r["content"] and "K" in r["content"]}

    # 3-4. Complete planner request: two example images + scene + crop, reasoning separated, decision parsed.
    pc = PromptConfig(allow_missing_examples=True)
    pb = PromptBuilder(pc)
    ex = args.examples or [args.crop, args.crop]
    pb._example_parts = _with_examples(pb, ex)
    clips = [f"C{i:02d}" for i in range(1, 11)]
    scene, crop = Image.open(args.scene).convert("RGB"), Image.open(args.crop).convert("RGB")
    state = PlannerState(crop_clip="C01")
    msgs = pb.messages(pb.task_text(clips, clips, state), scene, crop)
    rec = client.decide(msgs, clips, seed=2)
    a = rec.attempts[-1]
    report["full_request"] = {"failure": rec.failure, "decision": rec.decision and rec.decision.to_dict(),
                              "finish": a.get("finish_reason"), "usage": a.get("usage"), "latency_s": a.get("latency_s"),
                              "reasoning_chars": len(a.get("reasoning") or ""), "content_tail": (a.get("content") or "")[-300:],
                              "n_images": sum(p.get("type") == "image_url" for p in msgs[1]["content"]),
                              "retries": len(rec.attempts) - 1}

    # 5. Longest ten-clip history the budget allows (100 invocations).
    state = PlannerState(crop_clip="C10")
    skills = ["Insert", "Insert", "Pull", "Flatten"]
    state.history = [HistoryEntry(i + 1, skills[i % 4], clips[min(i // 10, 9)]) for i in range(100)]
    state.consecutive_inserts, state.counter_target = 2, "C10"
    msgs = pb.messages(pb.task_text(clips, clips, state), scene, crop)
    t = time.time()
    rec = client.decide(msgs, clips, seed=3)
    a = rec.attempts[-1]
    report["longest_history"] = {"failure": rec.failure, "decision": rec.decision and rec.decision.to_dict(),
                                 "finish": a.get("finish_reason"), "usage": a.get("usage"),
                                 "latency_s": time.time() - t, "gpu_memory_mib": gpu_memory(),
                                 "context_limit": 32768}

    # Optional: schema-constrained final output with reasoning enabled.
    if args.structured:
        sclient = PlannerClient(GenerationConfig(**{**gen.__dict__, "structured_output": True}))
        msgs = pb.messages(pb.task_text(clips, clips, PlannerState(crop_clip="C01")), scene, crop)
        rec = sclient.decide(msgs, clips, seed=4)
        a = rec.attempts[-1]
        report["structured_output"] = {"failure": rec.failure, "decision": rec.decision and rec.decision.to_dict(),
                                       "reasoning_chars": len(a.get("reasoning") or ""),
                                       "content": (a.get("content") or "")[:300], "finish": a.get("finish_reason")}

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("generation", "served_models")}, indent=1, default=str))


def _with_examples(pb: PromptBuilder, paths: list[str]) -> list[dict]:
    parts: list[dict] = [{"type": "text", "text": "## In-context examples"}]
    for ex, p in zip(pb.examples["image_examples"], paths):
        parts.append({"type": "image_url", "image_url": {"url": png_data_url(Image.open(p).convert("RGB"))}})
        parts.append({"type": "text", "text": ex["text"]})
    parts += [{"type": "text", "text": t} for t in pb.examples["verbal_examples"]]
    return parts


if __name__ == "__main__":
    main()
