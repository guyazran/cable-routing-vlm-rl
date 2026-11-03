"""Select the insertion checkpoint on the validation starts (deviation R10) and record the choice.

    python scripts/study/select_checkpoint.py --runs runs/study/train/sac_v6_s0 runs/study/train/sac_v6_s1 \
        --out reports/insertion_selection.json

Score = mean clean-success rate (success without collision) over the 100 single-clip and 100 route-stage
validation starts; ties go to fewer collisions, then to the earlier checkpoint. Only candidates with
both evaluations complete are eligible; incomplete ones are listed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(d: Path) -> list[dict]:
    return [json.loads(l) for f in sorted(d.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cands, incomplete = [], []
    for run in map(Path, args.runs):
        for mark in sorted(run.glob(".queued_*")):
            tag = mark.name.split("_", 2)[-1]
            name = f"{run.name}_{tag}"
            single, stage = rows(ROOT / "runs/study/insval" / f"{name}_single"), rows(ROOT / "runs/study/insval" / f"{name}_stage")
            policy = json.loads((ROOT / "runs/study/queue/insval/tasks" /
                                 f"p20_{name}_single_000.json").read_text())["policy"]
            m = re.search(r"sac_(\d+)_steps", policy)
            steps = int(m.group(1)) if m else json.loads((run / "train_meta.json").read_text())["env_steps"]
            if len(single) < 100 or len(stage) < 100:
                incomplete.append({"candidate": name, "n_single": len(single), "n_stage": len(stage)})
                continue
            clean = lambda rs: sum(r["success"] and not r["collided"] for r in rs)
            coll = sum(r["collided"] for r in single + stage)
            cands.append({"candidate": name, "policy": policy, "env_steps": steps,
                          "single_clean": clean(single), "stage_clean": clean(stage), "collisions": coll,
                          "score": (clean(single) + clean(stage)) / 200})
    cands.sort(key=lambda c: (-c["score"], c["collisions"], c["env_steps"]))
    report = {"rule": "max mean clean success over 100 single-clip + 100 route-stage validation starts "
                      "(seeds 8,000,000+ / 8,100,000+); ties: fewer collisions, then earlier checkpoint",
              "selected": cands[0] if cands else None, "candidates": cands, "incomplete": incomplete}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    for c in cands:
        print(f"{c['candidate']:24s} steps={c['env_steps']:6d} single={c['single_clean']:3d} stage={c['stage_clean']:3d} "
              f"coll={c['collisions']:2d} score={c['score']:.3f}")
    print("incomplete:", [i["candidate"] for i in incomplete])
    print("selected:", report["selected"]["candidate"] if cands else None)


if __name__ == "__main__":
    main()
