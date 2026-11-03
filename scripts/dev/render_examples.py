"""Render the planner's two in-context example images from DEVELOPMENT (train-split) scenes.

    CUDA_VISIBLE_DEVICES=3 python scripts/dev/render_examples.py --policy heuristic --out runs/dev/examples

For each candidate train scene: Insert(C01), then Pull(C02) (R6b: the head is carried to C02) with the
shared skills. Insert(C02) then runs primitive by primitive; the first crop centred on C02 is Example 1 (head about 4 cm in front of C02's entry, aligned, nothing
under the bridge). Insert(C02) then threads C02; the crop centred on C02 is Example 2 (cable under
the bridge, head just beyond the exit). A candidate is accepted only if the simulator's evaluator
confirms each state; every candidate's outcome is logged. Only train scenes are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cablerouting.cli import _load_state, _make_backend, _policy
from cablerouting.config import load_config
from cablerouting.scenes.generator import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="heuristic")
    ap.add_argument("--manifest", default="manifests/shapes_train.json")
    ap.add_argument("--shapes", default="arc,serpentine,spiral,circle")
    ap.add_argument("--n-clips", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    if "train" not in Path(args.manifest).name:
        raise SystemExit("examples must come from development (train) scenes")
    sim = _make_backend(cfg)
    from cablerouting.bridge import SimWorld, scene_spec
    from cablerouting.sim.routing_env import RoutingEnv
    from dloroute.decision import Decision
    from dloroute.skills import SkillLibrary
    from dloroute.geometry import tangent_yaw, wrap_angle
    from dloroute.rl.observation import observation
    from dloroute.rl.reward import insertion_success
    from dloroute.skills import decode_insert_action, insert_command
    from dloroute.state import head_metrics

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, scenes = load_manifest(Path(args.manifest))
    shapes = args.shapes.split(",")
    cands = [s for s in scenes if s.family in shapes and len(s.clips) == args.n_clips][: args.max_candidates]
    skills = SkillLibrary(_policy(cfg, args.policy), cfg.skills, cfg.obs, cfg.reward)
    log = []
    for scene in cands:
        fam = cfg.family(scene.family)
        env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
        env.reset(_load_state(scene.scene_id))
        world = SimWorld(env)
        rec = {"scene": scene.scene_id, "steps": []}

        def do(skill: str, clip: str):
            r = skills.execute(Decision(skill, clip, False), world)
            rs = env.routing(scene.route(2)).summary()
            rec["steps"].append({"skill": f"{skill}({clip})", "collided": r.collided, "routing": rs})
            return r, rs

        ok = False
        r, rs = do("Insert", "C01")
        if not r.collided and rs["sequence"][:1] == ["C01+"]:
            r, rs = do("Pull", "C02")
            if not r.collided:
                c02 = world.scene.clip("C02")
                m = head_metrics(c02, world.state())
                rec["c02_head"] = {"signed_dist": m.signed_dist, "rope_in": m.rope_in}
                if not r.collided and rs["sequence"] == ["C01+"] and not rs["partial"]:
                    # Insert(C02) primitive by primitive (as SkillLibrary.insert does); Example 1 is the
                    # first state with the head staged in front of C02: outside the clip, 2-8 cm before
                    # the entry face, within 1.5 cm of the axis and heading within 20 deg of the arrow.
                    full = crop1 = info1 = None
                    hits = 0
                    for t in range(cfg.skills.local_horizon):
                        st = world.state()
                        m = head_metrics(c02, st)
                        local = c02.pose.to_local(st.points[:1])[0]
                        yaw = float(wrap_angle(tangent_yaw(st.points, 1) - c02.pose.yaw))
                        rec.setdefault("insert_trace", []).append(
                            {"t": t, "head_local": local.tolist(), "rel_yaw_deg": float(np.degrees(yaw)),
                             "rope_in": m.rope_in})
                        if crop1 is None and not m.rope_in and -0.08 <= local[0] - c02.entry_x <= -0.02 \
                                and abs(local[1]) <= 0.015 and abs(yaw) <= np.radians(20):
                            full, crop1, info1 = env.render("C02")
                        a = np.asarray(skills.policy(observation(st.points, c02, m.rope_in, cfg.obs)), dtype=float)
                        act = decode_insert_action(a, c02, cfg.obs.n_points, cfg.skills)
                        res = world.pick_place(insert_command(act, c02, cfg.obs,
                                                              world.scene.cable_radius + cfg.skills.insert_clearance))
                        if res.collided or insertion_success(head_metrics(c02, world.state()), cfg.reward):
                            hits = int(res.collided)
                            break
                    rs = env.routing(scene.route(2)).summary()
                    rec["steps"].append({"skill": "Insert(C02) stepwise", "collided": bool(hits), "routing": rs})
                    if crop1 is not None and not hits and rs["sequence"] == ["C01+", "C02+"]:
                        _, crop2, info2 = env.render("C02")
                        crop1.save(out / f"{scene.scene_id}_insert_example.png")
                        crop2.save(out / f"{scene.scene_id}_pull_example.png")
                        full.save(out / f"{scene.scene_id}_full_before_insert.png")
                        rec.update({"accepted": True, "crop_bounds": [info1["bounds"], info2["bounds"]]})
                        ok = True
        rec.setdefault("accepted", False)
        log.append(rec)
        print(json.dumps({k: rec[k] for k in ("scene", "accepted")} | {"last": rec["steps"][-1]["skill"]}), flush=True)
        if ok:
            break
    (out / "candidates.json").write_text(json.dumps(log, indent=1, default=str))
    sim.close()


if __name__ == "__main__":
    main()
