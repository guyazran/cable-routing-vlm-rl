"""Simulator speed benchmark on this node (physics rate, render FPS, skill wall times).

    CUDA_VISIBLE_DEVICES=2 python scripts/bench/sim_speed.py --out runs/bench/full.json
    CUDA_VISIBLE_DEVICES=2 python scripts/bench/sim_speed.py --quick --out runs/bench/quick_a.json   # concurrency runs

Uses a development scene generated in memory; nothing here is study data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np


def now() -> float:
    return time.perf_counter()


def sysinfo() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    try:
        gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        gov = "?"
    return {"gpus_visible": gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu": cpu,
            "cpu_governor": gov, "cpus_allocated": len(os.sched_getaffinity(0)), "python": platform.python_version()}


def gpu_mem_mib() -> int | None:
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    R: dict = {"sys": sysinfo(), "quick": args.quick}
    t_wall0 = now()

    from cablerouting.config import load_config
    from cablerouting.scenes.generator import generate_pool

    cfg = load_config()
    fam = cfg.family("primary10")
    scene = generate_pool(fam, "development", 1)[0][0]

    t = now()
    from cablerouting.sim.isaac_backend import IsaacBackend

    sim = IsaacBackend(cfg.cable, cfg.physics, render_resolutions=(cfg.render.full_resolution,
                                                                   (cfg.render.crop_resolution,) * 2))
    R["startup_s"] = now() - t

    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    from cablerouting.sim.isaac_backend import _mat_to_quat_wxyz
    from cablerouting.bridge import SimWorld
    from cablerouting.sim.routing_env import RoutingEnv
    from cablerouting.robot.executor import PickPlace, Via
    from dloroute.decision import Decision
    from dloroute.rl.policies import HeuristicInsertPolicy
    from dloroute.skills import SkillLibrary

    t = now()
    env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
    R["first_load_scene_s"] = now() - t
    sim.set_polyline(scene.initial_polyline(cfg.cable.link_length))
    sim.step(480)
    state = sim.capture_state()
    dt = sim.physics_dt
    R["physics_dt_s"] = dt
    R["n_cable_links"] = cfg.cable.n_links

    # ---------------------------------------------------------------- physics
    phys = {}
    for k in range(5):  # batched stepping, cable at rest
        t = now()
        sim.step(240)
        phys.setdefault("batched_rest_steps_per_s", []).append(240 / (now() - t))
    env.reset(state)
    t = now()
    for _ in range(480):
        sim.step(1)
    phys["loop_rest_steps_per_s"] = 480 / (now() - t)

    # Held, moving cable: attach near the head and drag 10 cm at 0.1 m/s, stepping one at a time.
    env.reset(state)
    pts = sim.cable_points()
    c1 = env.clips[0]
    yaw = float(np.arctan2(*(pts[1] - pts[3])[1::-1]))
    start = np.array([*pts[2], yaw])
    sim.set_gripper_pose(start, 0.012)
    sim.attach(sim.point_segment(2))
    n = int(round(1.0 / dt))
    t = now()
    for i in range(1, n + 1):
        tgt = start.copy()
        tgt[:2] += i / n * 0.10 * c1.pose.heading
        sim.step(1, gripper_target=tgt)
    phys["loop_held_moving_steps_per_s"] = n / (now() - t)
    sim.detach()

    # Full primitive through the executor (collision checks every step, settle, free moves).
    times = []
    for _ in range(3):
        env.reset(state)
        p0 = sim.cable_points()
        goal = c1.pose.to_world(np.array([c1.pose.to_local(p0[2])[0] + 0.06, 0.0]))
        t = now()
        r = env.executor.pick_place(PickPlace(2, [Via(goal[0], goal[1], c1.pose.yaw)], scene.cable_radius + 0.001, "b"))
        times.append((now() - t, r.physics_steps))
    phys["pick_place_wall_s"] = [w for w, _ in times]
    phys["pick_place_physics_steps"] = times[0][1]
    phys["pick_place_steps_per_s"] = float(np.mean([s / w for w, s in times]))
    for key in ("batched_rest_steps_per_s",):
        phys[key] = float(np.median(phys[key]))
    phys["realtime_factor"] = {k: v * dt for k, v in phys.items() if k.endswith("steps_per_s")}
    R["physics"] = phys

    # ---------------------------------------------------------------- rendering
    def make_cam(name: str, res: tuple[int, int]):
        cam = UsdGeom.Camera.Define(sim.stage, f"/World/Cameras/bench_{name}")
        cam.CreateFocalLengthAttr(22.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * res[1] / res[0])
        cam.CreateClippingRangeAttr((0.01, 50.0))
        eye = np.array([*(c1.pose.xy + np.array([0.25, 0.25])), 0.35])
        tgt = np.array([*c1.pose.xy, 0.0])
        f = (tgt - eye) / np.linalg.norm(tgt - eye)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        q = _mat_to_quat_wxyz(np.column_stack([right, np.cross(right, f), -f]))
        xf = UsdGeom.Xformable(cam)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, eye)))
        xf.AddOrientOp().Set(Gf.Quatf(*map(float, q)))
        rp = rep.create.render_product(cam.GetPath().pathString, res)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        return rp, ann

    products = {name: (rp, ann) for name, (_, rp, ann, _) in sim._cams.items()}
    vres = {"360p": (640, 360), "720p": (1280, 720)} if args.quick else \
        {"360p": (640, 360), "720p": (1280, 720), "1080p": (1920, 1080)}
    for name, res in vres.items():
        products[name] = make_cam(name, res)

    def only(*names) -> None:
        for nm, (rp, _) in products.items():
            rp.hydra_texture.set_updates_enabled(nm in names)

    def fps(names, sub: int, frames: int = 20, warm: int = 3) -> float:
        only(*names)
        ann = products[names[0]][1]
        for _ in range(warm):
            rep.orchestrator.step(rt_subframes=sub, pause_timeline=True, delta_time=0.0)
        t = now()
        for _ in range(frames):
            sim._sync_render()
            rep.orchestrator.step(rt_subframes=sub, pause_timeline=True, delta_time=0.0)
            np.asarray(ann.get_data())
        return frames / (now() - t)

    rend = {}
    subs = (2,) if args.quick else (1, 2, 4)
    for name in vres:
        for s in subs:
            rend[f"{name}_rt{s}"] = fps([name], s)
    if not args.quick:
        rend["planner_full_1600x1200_rt4"] = fps(["full"], 4)
        rend["planner_zoom_768x768_rt4"] = fps(["zoom"], 4)
        rend["planner_full+zoom_one_step_rt4"] = fps(["full", "zoom"], 4)
        rend["720p_rt2_with_all_products_active"] = fps(list(products), 2)
        # The planner observation as used by the episode loop: env.render renders both cameras.
        only("full", "zoom")
        t = now()
        for _ in range(5):
            env.render("C01")
        rend["planner_env_render_obs_per_s"] = 5 / (now() - t)
    R["render_fps"] = rend

    # ---------------------------------------------------------------- recording loop
    only("720p")
    ann = products["720p"][1]
    env.reset(state)
    p0 = sim.cable_points()
    start = np.array([*p0[2], float(np.arctan2(*(p0[1] - p0[3])[1::-1]))])
    sim.set_gripper_pose(start, 0.012)
    sim.attach(sim.point_segment(2))
    every = max(1, round(1 / (30 * dt)))
    frames = 60
    t = now()
    for i in range(frames * every):
        tgt = start.copy()
        tgt[:2] += (i + 1) / (frames * every) * 0.08 * c1.pose.heading
        sim.step(1, gripper_target=tgt)
        if (i + 1) % every == 0:
            sim._sync_render()
            rep.orchestrator.step(rt_subframes=2, pause_timeline=True, delta_time=0.0)
            np.asarray(ann.get_data())
    w = now() - t
    sim.detach()
    R["recording_720p_rt2"] = {"frames": frames, "physics_steps_per_frame": every, "output_fps": frames / w,
                               "video_seconds_per_wall_second": frames / 30 / w}

    # ---------------------------------------------------------------- skills and scene ops
    if not args.quick:
        only("full", "zoom")
        ops = {}
        t = now()
        for _ in range(3):
            sim.load_scene(env.clips)
        ops["load_scene_s"] = (now() - t) / 3
        t = now()
        for _ in range(20):
            env.reset(state)
        ops["restore_state_s"] = (now() - t) / 20
        library = SkillLibrary(HeuristicInsertPolicy(cfg.obs), cfg.skills, cfg.obs, cfg.reward)
        world = SimWorld(env)
        skills = {}
        for label, decisions in (("insert_C01", [Decision("Insert", "C01", False)]),
                                 ("insert_C01_then_pull_C01", [Decision("Insert", "C01", False),
                                                               Decision("Pull", "C01", False)]),
                                 ("flatten_C01", [Decision("Flatten", "C01", False)])):
            env.reset(state)
            rows = []
            for d in decisions:
                t = now()
                res = library.execute(d, world)
                rows.append({"skill": f"{d.skill}({d.target_clip})", "wall_s": now() - t,
                             "pick_places": res.physical_ops, "collided": res.collided})
            skills[label] = rows
        R["scene_ops"] = ops
        R["skills"] = skills

    R["gpu_mem_used_mib"] = gpu_mem_mib()
    R["max_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    R["total_wall_s"] = now() - t_wall0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1, default=float))
    print("[bench] " + json.dumps(R, default=float), flush=True)
    sim.close()


if __name__ == "__main__":
    main()
