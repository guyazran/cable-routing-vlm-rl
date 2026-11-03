"""Per-process speed and CPU use of the insertion training environment under thread limits.

    CABLE_SIM_THREADS=2 CABLE_PHYSX_THREADS=1 CUDA_VISIBLE_DEVICES=3 python scripts/bench/thread_bench.py --out runs/bench/threads_2_1.json

Runs a fixed sequence of insertion-task resets and random primitives (fixed seeds) and reports wall
time, process CPU time and a hash of the final cable state, so runs under different thread settings
can be compared for speed and for bitwise-identical physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from cablerouting.config import load_config
from cablerouting.cli import _insertion_world, _make_backend, _stage_source


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--progress", action="store_true", help="print a line per reset (leak tests)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--cameras", action="store_true", help="build the camera render products (routing setup)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.perf_counter()
    if args.cameras:  # full-size viewport and render products, as in routing episodes
        os.environ["CABLE_INSERT_VIEW"] = "1280x720"
    sim = _make_backend(cfg, robot=False)
    import carb

    settings = carb.settings.get_settings()
    startup = time.perf_counter() - t0
    from dloroute.rl.training import InsertionEnv

    stages, _ = _stage_source("manifests/shapes_train.json")
    env = InsertionEnv(_insertion_world(cfg, sim, "primary10", 0.5, stages), cfg.skills, cfg.obs, cfg.reward, seed=0)
    rng = np.random.default_rng(0)
    resets, steps, h = [], [], hashlib.sha256()
    c0, w0 = cpu_s(), time.perf_counter()
    for ep in range(args.episodes):
        t = time.perf_counter()
        env.reset(seed=1000 + ep)
        resets.append(time.perf_counter() - t)
        if args.progress:
            print(f"[reset] {ep + 1}", flush=True)
        for _ in range(args.steps):
            t = time.perf_counter()
            _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 7).astype(np.float32))
            steps.append(time.perf_counter() - t)
            h.update(np.ascontiguousarray(sim.cable_points()).tobytes())
            if term or trunc:
                break
    wall, cpu = time.perf_counter() - w0, cpu_s() - c0
    rec = {"CABLE_SIM_THREADS": os.environ.get("CABLE_SIM_THREADS"),
           "CABLE_PHYSX_THREADS": os.environ.get("CABLE_PHYSX_THREADS"),
           "physx_num_threads": settings.get("/persistent/physics/numThreads"),
           "tasking_threads": settings.get("/plugins/carb.tasking.plugin/threadCount"),
           "rtx_reserved_descriptors": settings.get("/rtx/reservedDescriptors"),
           "rtx_descriptor_sets": settings.get("/rtx/descriptorSets"),
           "startup_s": startup, "reset_s_mean": float(np.mean(resets)), "step_s_mean": float(np.mean(steps)),
           "n_steps": len(steps), "wall_s": wall, "cpu_s": cpu, "cores_busy": cpu / wall,
           "state_hash": h.hexdigest()[:16]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print("[thread-bench]", json.dumps(rec), flush=True)
    sim.close()


if __name__ == "__main__":
    main()
