"""Does rendering advance or perturb physics? Compare cable states with and without renders.

    CUDA_VISIBLE_DEVICES=1 python scripts/smoke/render_physics_check.py
"""

from __future__ import annotations

import json

import numpy as np

from cablerouting.cli import _load_state, _make_backend
from cablerouting.config import ROOT, load_config
from cablerouting.scenes.generator import load_manifest


def main() -> None:
    cfg = load_config()
    _, scenes = load_manifest(ROOT / "manifests" / "hairpin6_development.json")
    scene = scenes[0]
    sim = _make_backend(cfg)
    from cablerouting.sim.video_camera import VideoCamera, capture_all

    sim.load_scene(scene.clip_objects())
    state = _load_state(scene.scene_id)
    cam = VideoCamera(sim, "check", (640, 360))
    cam.look(np.array([1.5, -1.0, 1.2]), np.array([0.3, 0.0, 0.0]))
    pts = None

    def episode(render_every: int | None, label: str) -> np.ndarray:
        sim.restore_state(state)
        i = 40  # grasp a point 0.5 m behind the head and drag it in a circle
        p = sim.cable_points()[i]
        sim.set_gripper_pose(np.array([p[0], p[1], p[2], 0.0]), 0.012)
        sim.attach(sim.point_segment(i))
        t0 = sim._SM.get_simulation_time()
        for k in range(480):
            a = 2 * np.pi * k / 480
            tgt = np.array([p[0] + 0.08 * np.sin(a), p[1] + 0.08 * (1 - np.cos(a)), p[2] + 0.03 * np.sin(a / 2), 0.0])
            sim.step(1, gripper_target=tgt)
            if render_every and k % render_every == 0:
                capture_all(sim, [cam])
        sim.detach()
        sim.step(120)
        out = sim.cable_points().copy()
        print(f"[check] {label}: sim time advanced {sim._SM.get_simulation_time() - t0:.4f} s "
              f"for {480 + 120} steps ({600 * sim.physics_dt:.4f} s)", flush=True)
        return out

    a = episode(None, "no render A")
    b = episode(None, "no render B")
    c = episode(8, "render every 8 steps")
    d = episode(None, "no render C (after rendering)")
    res = {"A_vs_B_max_m": float(np.abs(a - b).max()), "A_vs_render_max_m": float(np.abs(a - c).max()),
           "A_vs_C_max_m": float(np.abs(a - d).max())}
    print("[check]", json.dumps(res), flush=True)
    (ROOT / "runs" / "smoke" / "render_steps" / "result.json").write_text(json.dumps(res, indent=1))
    sim.close()


if __name__ == "__main__":
    main()
