"""Put study units on a shared work queue (src/cablerouting/eval/workqueue.py).

    python scripts/study/enqueue.py insertion-eval --queue Q --policy CKPT --seed-start 8000000 --n 100 \
        --chunk 10 --route-stage-prob 0.5 --stages manifests/shapes_train.json --out runs/study/insval/NAME
    python scripts/study/enqueue.py episodes --queue Q --manifest M --scenes ID[,ID...] | --all [--n-clips 10] \
        --controllers reference --targets 10 --policy P --budget 10 --out runs/study/dev/NAME
    python scripts/study/enqueue.py schedule --queue Q --manifest M --controllers vlm,reference --targets 5,10 \
        --policy P --out runs/study/screen     # protocol Section 10: randomized condition order per scene

``schedule`` follows ``cablerouting run-screen``: one episode seed per scene (1000 * scene index), shared by
its conditions, and the condition order within each scene permuted with the study's scheduling seed. The
task ids carry the schedule position, so workers take episodes in schedule order. Enqueuing is idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cablerouting.config import ROOT, load_config
from cablerouting.eval.workqueue import Queue
from cablerouting.scenes.generator import load_manifest


def rel(p: str | Path) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def cmd_insertion_eval(a) -> None:
    q = Queue(Path(rel(a.queue)))
    out = Path(rel(a.out))
    n_new = 0
    for k, s in enumerate(range(a.seed_start, a.seed_start + a.n, a.chunk)):
        spec = {"kind": "insertion_eval", "policy": rel(a.policy) if a.policy != "heuristic" else a.policy,
                "seed_start": s, "n": min(a.chunk, a.seed_start + a.n - s), "route_stage_prob": a.route_stage_prob,
                "stages": a.stages, "scale": a.scale, "out": str(out / f"chunk_{k:03d}.jsonl")}
        n_new += q.add(f"{a.priority}_{out.name}_{k:03d}", spec)
    print(f"{n_new} insertion-eval tasks added to {q.root}")


def _episode_spec(manifest: str, scene_id: str, controller: str, targets: int, policy: str, budget: int,
                  seed: int, out: Path, overrides: dict | None = None) -> dict:
    spec = {"kind": "episode", "manifest": rel(manifest), "scene_id": scene_id, "controller": controller,
            "targets": targets, "policy": rel(policy) if policy != "heuristic" else policy, "budget": budget,
            "seed": seed, "out": str(out)}
    if overrides:
        spec["overrides"] = overrides
    return spec


def cmd_episodes(a) -> None:
    q = Queue(Path(rel(a.queue)))
    _, scenes = load_manifest(Path(rel(a.manifest)))
    if a.scenes:
        want = a.scenes.split(",")
        scenes = [s for s in scenes if s.scene_id in want]
    elif a.n_clips:
        scenes = [s for s in scenes if len(s.clips) == a.n_clips]
    if a.limit:
        scenes = scenes[: a.limit]
    out = Path(rel(a.out))
    n_new = 0
    for i, s in enumerate(scenes):
        for c in a.controllers.split(","):
            for h in map(int, a.targets.split(",")):
                spec = _episode_spec(a.manifest, s.scene_id, c, h, a.policy, a.budget, a.seed + 1000 * i,
                                     out / s.scene_id / f"{c}_{h}", json.loads(a.overrides) if a.overrides else None)
                n_new += q.add(f"{a.priority}_{out.name}_{i:04d}_{c}_{h}", spec)
    print(f"{n_new} episode tasks added to {q.root} ({len(scenes)} scenes)")


def cmd_schedule(a) -> None:
    cfg = load_config()
    q = Queue(Path(rel(a.queue)))
    _, scenes = load_manifest(Path(rel(a.manifest)))
    conditions = [(c, h) for c in a.controllers.split(",") for h in map(int, a.targets.split(","))]
    rng = np.random.default_rng(cfg.study.scheduling_seed)
    out = Path(rel(a.out))
    out.mkdir(parents=True, exist_ok=True)
    schedule, n_new = [], 0
    for i, s in enumerate(scenes):
        order = rng.permutation(len(conditions))
        for j in order:
            c, h = conditions[j]
            pos = len(schedule)
            schedule.append({"position": pos, "scene": s.scene_id, "controller": c, "targets": h,
                             "episode_seed": 1000 * i})
            spec = _episode_spec(a.manifest, s.scene_id, c, h, a.policy, cfg.study.per_clip_budget, 1000 * i,
                                 out / s.scene_id / f"{c}_{h}")
            n_new += q.add(f"{a.priority}_{out.name}_{pos:04d}", spec)
    sched = out / "schedule.json"
    if not sched.exists():
        sched.write_text(json.dumps({"scheduling_seed": cfg.study.scheduling_seed, "manifest": a.manifest,
                                     "conditions": conditions, "episodes": schedule}, indent=1))
    print(f"{n_new} scheduled episodes added to {q.root}; schedule in {sched}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("insertion-eval")
    p.add_argument("--queue", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--route-stage-prob", type=float, default=0.0)
    p.add_argument("--stages", default="none")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--priority", default="p50")
    p.add_argument("--out", required=True)
    p = sub.add_parser("episodes")
    p.add_argument("--queue", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--scenes")
    p.add_argument("--n-clips", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--controllers", default="reference")
    p.add_argument("--targets", default="10")
    p.add_argument("--policy", required=True)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overrides", help='development only: JSON {"dotted.key": value} config overrides')
    p.add_argument("--priority", default="p50")
    p.add_argument("--out", required=True)
    p = sub.add_parser("schedule")
    p.add_argument("--queue", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--controllers", default="vlm,reference")
    p.add_argument("--targets", default="5,10")
    p.add_argument("--policy", required=True)
    p.add_argument("--priority", default="p10")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"insertion-eval": cmd_insertion_eval, "episodes": cmd_episodes, "schedule": cmd_schedule}[a.cmd](a)


if __name__ == "__main__":
    main()
