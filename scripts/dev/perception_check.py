"""Perception check on development images (protocol Section 4): does the planner model recognise the
threading state of the clip in a zoomed view?

    python scripts/dev/perception_check.py --runs runs/study/dev --n 60 --out runs/dev/perception

Images come from development episode traces (train-split scenes only): every decision stored the crop
centred on one clip and the simulator's routing state before the decision. Ground truth for the crop's
clip: "threaded" (a forward transit), "partial" (entered, not out) or "not" (neither). Crops are sampled
balanced over the three labels where possible, with a fixed seed. The model gets the same system prompt
context about the scene, the crop at the study's resolution, and a direct question; sampling settings
are the frozen planner profile. Reports accuracy, the confusion matrix, and the rate of calling a clip
threaded when it is not (the error that leads to premature completion).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from cablerouting.config import load_config
from dloroute.planner.prompts import png_data_url

QUESTION = (
    "This is a zoomed-in top-down view centred on clip {clip}. The cable is red; clips are white omega-shaped "
    "brackets with a raised bridge, a printed label and a blue arrow giving the insertion direction.\n"
    "Question: what is the threading state of clip {clip}?\n"
    "- \"threaded\": the cable passes under {clip}'s bridge from its entry side and comes out on its exit side;\n"
    "- \"partial\": the cable head has entered under the bridge but has not come out on the exit side;\n"
    "- \"not\": no cable passes under {clip}'s bridge.\n"
    "Reason briefly, then end with exactly one JSON object: {{\"state\": \"threaded\" | \"partial\" | \"not\"}}")


def collect(runs: Path) -> list[dict]:
    items = []
    for trace in sorted(runs.glob("*/*/*/trace.jsonl")):
        ep = trace.parent
        for line in trace.read_text().splitlines():
            d = json.loads(line)
            crop_rel = (d.get("images") or {}).get("crop")
            info = d.get("info") or {}
            clip = (info.get("crop") or {}).get("clip")
            rb = d.get("routing_before")
            if not crop_rel or not clip or rb is None:
                continue
            seq = rb.get("sequence", [])
            partial = rb.get("partial") or {}
            label = "threaded" if f"{clip}+" in seq else ("partial" if clip in partial else "not")
            items.append({"image": str(ep / crop_rel), "clip": clip, "label": label,
                          "scene": ep.parent.name, "episode": str(ep)})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/study/dev")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--base-url")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    g = cfg.generation
    items = collect(Path(args.runs))
    rng = random.Random(args.seed)
    by = {k: [i for i in items if i["label"] == k] for k in ("threaded", "partial", "not")}
    per = args.n // 3
    chosen = []
    for k, v in by.items():
        rng.shuffle(v)
        chosen += v[:per]
    rest = [i for i in items if i not in chosen]
    rng.shuffle(rest)
    chosen += rest[: args.n - len(chosen)]
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url or g.base_url, api_key="EMPTY", timeout=g.timeout_s, max_retries=0)

    def ask(k_item):
        k, it = k_item
        img = Image.open(it["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_data_url(img)}},
                                             {"type": "text", "text": QUESTION.format(clip=it["clip"])}]}]
        r = client.chat.completions.create(
            model=g.model, messages=msgs, temperature=g.temperature, top_p=g.top_p, max_tokens=g.max_tokens,
            seed=1000 + k, reasoning_effort=g.reasoning_effort,
            extra_body={"top_k": g.top_k, "min_p": g.min_p,
                        "chat_template_kwargs": {"enable_thinking": g.enable_thinking}})
        text = r.choices[0].message.content or ""
        m = re.findall(r'"state"\s*:\s*"(threaded|partial|not)"', text)
        return {**it, "answer": m[-1] if m else None, "finish": r.choices[0].finish_reason,
                "completion_tokens": r.usage.completion_tokens if r.usage else None}

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(ask, enumerate(chosen)))
    conf = Counter((r["label"], r["answer"]) for r in res)
    acc = sum(r["label"] == r["answer"] for r in res) / len(res)
    false_threaded = [r for r in res if r["answer"] == "threaded" and r["label"] != "threaded"]
    n_not_threaded = sum(r["label"] != "threaded" for r in res)
    summary = {"n": len(res), "available": {k: len(v) for k, v in by.items()}, "accuracy": acc,
               "confusion": {f"{a}->{b}": c for (a, b), c in sorted(conf.items(), key=str)},
               "false_threaded_rate": len(false_threaded) / max(1, n_not_threaded),
               "unparsed": sum(r["answer"] is None for r in res), "model": g.model,
               "note": "development images only (train-split scenes); not evaluation data"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "items": res}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
