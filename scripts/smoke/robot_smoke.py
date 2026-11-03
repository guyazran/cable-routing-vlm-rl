"""Robot smoke check: the Franka loads, matches the kinematic model, follows the grasp
body, and is hidden from observation renders.

    CUDA_VISIBLE_DEVICES=3 python scripts/smoke/robot_smoke.py --out runs/smoke/robot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cablerouting.cli import _load_state, _make_backend
from cablerouting.config import ROOT, load_config
from cablerouting.scenes.generator import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke/robot")
    ap.add_argument("--scene", default="primary10_development:13")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    man, idx = args.scene.split(":")
    _, scenes = load_manifest(ROOT / "manifests" / f"{man}.json")
    scene = scenes[int(idx)]
    sim = _make_backend(cfg)
    from cablerouting.sim.routing_env import RoutingEnv
    from cablerouting.sim.video_camera import VideoCamera

    fam = cfg.family(scene.family)
    env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
    env.reset(_load_state(scene.scene_id))
    cam = VideoCamera(sim, "wide", (1280, 720), focal_mm=18.0)
    cam.look(np.array([1.55, -1.25, 1.25]), np.array([0.25, 0.0, 0.12]))
    report = {"fk_check_m": sim.robot.fk_check, "fingertip_depth": sim.robot.fingertip_depth,
              "base_box": [v.tolist() for v in sim.robot.base_box], "poses": []}
    c1 = env.clips[0]
    poses = {"home": np.array(cfg.motion.home_pose),
             "above_C01": np.array([*c1.pose.xy, 0.08, c1.pose.yaw]),
             "at_C01": np.array([*(c1.pose.xy - 0.05 * c1.pose.heading), 0.005, c1.pose.yaw + 0.4])}
    for name, pose in poses.items():
        sim.set_gripper_pose(pose, 0.012 if name == "at_C01" else 0.08)
        Image.fromarray(cam.capture()).save(out / f"wide_{name}.png")
        r = sim.robot.last
        report["poses"].append({"name": name, "pose": pose.tolist(), "q": r.q.tolist(),
                                "pos_err_mm": 1e3 * r.pos_err, "rot_err_deg": float(np.degrees(r.rot_err))})
        print("[robot]", report["poses"][-1], flush=True)
    close = VideoCamera(sim, "close", (640, 720), focal_mm=18.0)
    close.look(np.array([*(c1.pose.xy - 0.25 * c1.pose.heading + 0.2 * np.array([-c1.pose.heading[1], c1.pose.heading[0]])), 0.22]),
               np.array([*c1.pose.xy, 0.02]))
    Image.fromarray(close.capture()).save(out / "close_at_C01.png")
    full, crop, _ = env.render("C01")
    for name, img in (("observation_full", full), ("observation_crop", crop)):
        (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(out / f"{name}.png")
    (out / "robot_smoke.json").write_text(json.dumps(report, indent=1))
    print("[robot] ok", json.dumps({k: report[k] for k in ("fk_check_m", "fingertip_depth", "base_box")}), flush=True)
    sim.close()


if __name__ == "__main__":
    main()
