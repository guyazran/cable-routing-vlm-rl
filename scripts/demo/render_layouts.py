"""Render full top-down views of development-scene layouts from their saved initial states.

    CUDA_VISIBLE_DEVICES=2 python scripts/demo/render_layouts.py --out runs/demo/layouts \
        --scenes primary10_development:0,1,2,3,4,5 paper5_development:0,1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cablerouting.cli import _load_state, _make_backend
from cablerouting.config import ROOT, load_config
from cablerouting.scenes.generator import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo/layouts")
    ap.add_argument("--scenes", nargs="+", default=["primary10_development:0,1,2,3,4,5"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    sim = _make_backend(cfg)
    from cablerouting.sim.routing_env import RoutingEnv

    meta = []
    for spec in args.scenes:
        name, idx = spec.split(":")
        _, scenes = load_manifest(ROOT / "manifests" / f"{name}.json")
        for i in map(int, idx.split(",")):
            s = scenes[i]
            fam = cfg.family(s.family)
            env = RoutingEnv(sim, s, fam.view_x, fam.view_y, cfg.motion, cfg.render)
            env.reset(_load_state(s.scene_id))
            full, _, _ = env.render(s.clips[0]["clip_id"])
            full.save(out / f"{s.scene_id}.png")
            meta.append({"scene_id": s.scene_id, "family": s.family, "n_clips": len(s.clips), "variant": s.variant,
                         "seed": s.seed, "turns_deg": [round(t["relative_angle_deg"], 1) for t in s.stages],
                         "chords_m": [round(t["chord"], 3) for t in s.stages],
                         "route_arc_m": round(s.route_arc_length, 3), "route_slack_m": round(s.route_slack, 3),
                         "reach_margin_m": round(s.reach_margin, 3)})
            print(f"[layouts] {s.scene_id}", flush=True)
    (out / "layouts.json").write_text(json.dumps(meta, indent=1))
    sim.close()


if __name__ == "__main__":
    main()
