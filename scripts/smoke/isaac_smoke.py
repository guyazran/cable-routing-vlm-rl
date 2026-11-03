"""End-to-end simulator smoke check (not an experiment).

Builds a development scene of the ten-clip family, settles the cable, checks
cable integrity, state save/restore, rendering with labels, one pick-and-place,
reset determinism, and the threading evaluator on a cable laid through C01.

    conda activate cable
    CUDA_VISIBLE_DEVICES=3 python scripts/smoke/isaac_smoke.py --out runs/smoke/isaac
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke/isaac")
    ap.add_argument("--family", default="primary10")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    from cablerouting.config import load_config
    from cablerouting.scenes.generator import generate_pool

    cfg = load_config()
    fam = cfg.family(args.family)
    scenes, _ = generate_pool(fam, "development", args.index + 1)
    scene = scenes[args.index]
    report["scene_id"] = scene.scene_id

    t0 = time.time()
    from cablerouting.sim.isaac_backend import IsaacBackend

    sim = IsaacBackend(cfg.cable, cfg.physics, render_resolutions=(cfg.render.full_resolution,
                                                                   (cfg.render.crop_resolution,) * 2))
    report["startup_s"] = time.time() - t0

    from cablerouting.geometry.threading import evaluate_routing
    from cablerouting.sim.routing_env import RoutingEnv
    from cablerouting.robot.executor import PickPlace, Via

    env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
    clips = env.clips

    def integrity(tag: str) -> dict:
        p = sim.cable_points()
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        d = {"n_points": len(p), "seg_mean": float(seg.mean()), "seg_max_stretch": float(seg.max() / cfg.cable.link_length - 1),
             "z_min": float(p[:, 2].min()), "z_max": float(p[:, 2].max()), "head": p[0].round(4).tolist(),
             "length": float(seg.sum())}
        report[f"integrity_{tag}"] = d
        print(f"[smoke] {tag}: {json.dumps(d)}", flush=True)
        return d

    # 1. Initial state: nominal straight cable, settled.
    sim.set_polyline(scene.initial_polyline(cfg.cable.link_length))
    t = time.time()
    sim.step(480)
    report["settle_480_steps_s"] = time.time() - t
    integrity("settled")
    init_state = sim.capture_state()
    np.savez_compressed(out / "initial_state.npz", **{k: v for k, v in init_state.items() if isinstance(v, np.ndarray)})

    # 2. Rendering with labels.
    t = time.time()
    full, crop, info = env.render("C01")
    report["render_both_s"] = time.time() - t
    full.save(out / "full.png")
    crop.save(out / "crop_C01.png")
    report["crop_info"] = info
    t = time.time()
    env.render("C02")
    report["render_both_warm_s"] = time.time() - t

    # 3. One pick-and-place: drag the head 6 cm along C01's axis (at drag height).
    c1 = clips[0]
    p0 = sim.cable_points()
    goal = c1.pose.to_world(np.array([c1.pose.to_local(p0[2])[0] + 0.06, 0.0]))
    plan = PickPlace(2, [Via(goal[0], goal[1], c1.pose.yaw)], scene.cable_radius + 0.001, "smoke-drag")
    t = time.time()
    res = env.executor.pick_place(plan)
    report["pick_place_s"] = time.time() - t
    report["pick_place"] = {k: v for k, v in res.to_dict().items() if k != "trajectory"}
    integrity("after_drag")
    p1 = sim.cable_points()
    report["head_displacement"] = float(np.linalg.norm(p1[0, :2] - p0[0, :2]))
    full, crop, _ = env.render("C01")
    full.save(out / "full_after_drag.png")
    crop.save(out / "crop_after_drag.png")

    # 4. Reset determinism: restore, repeat the same primitive, compare.
    finals = []
    for _ in range(2):
        env.reset(init_state)
        env.executor.pick_place(PickPlace(2, [Via(goal[0], goal[1], c1.pose.yaw)], scene.cable_radius + 0.001, "rep"))
        finals.append(sim.cable_points())
    report["restore_repeat_max_diff_m"] = float(np.abs(finals[0] - finals[1]).max())
    report["pick_place_warm_s"] = None
    env.reset(init_state)
    t = time.time()
    r2 = env.executor.pick_place(PickPlace(2, [Via(goal[0], goal[1], c1.pose.yaw)], scene.cable_radius + 0.001, "warm"))
    report["pick_place_warm_s"] = time.time() - t
    report["pick_place_warm_steps"] = r2.physics_steps
    env.reset(init_state)
    report["restore_exact_max_diff_m"] = float(np.abs(sim.cable_points() - p0).max())

    # 5. Evaluator on a cable laid straight through C01 (head 5 cm past its exit).
    n = cfg.cable.n_links
    head = c1.pose.to_world(np.array([c1.exit_x + 0.05, 0.0]))
    s = np.arange(n + 1) * cfg.cable.link_length
    poly = np.column_stack([head - s[:, None] * c1.pose.heading, np.full(n + 1, cfg.cable.radius)])
    sim.set_polyline(poly)
    sim.step(240)
    r = evaluate_routing(clips, sim.cable_points(), scene.route(5))
    report["threaded_C01"] = r.summary()
    integrity("threaded_C01")
    full, crop, _ = env.render("C01")
    crop.save(out / "crop_threaded_C01.png")

    # 6. Throughput.
    t = time.time()
    sim.step(240)
    report["physics_240_steps_s"] = time.time() - t

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("[smoke] REPORT " + json.dumps(report, default=str), flush=True)
    sim.close()


if __name__ == "__main__":
    main()
