"""Plumbing check of the insertion training environment (no training, no checkpoint kept).

    python scripts/smoke/train_env_smoke.py [stage manifest | none]   (default: the official train split)

Also starts route-stage tasks from the sharpest stages (|turn| > 80 deg) and checks the start
state physically: the cable is threaded through the preceding clip P only.
"""
import json
import sys
import time

import numpy as np

from cablerouting.config import load_config
from cablerouting.sim.isaac_backend import IsaacBackend, RobotSpec

cfg = load_config()
# No robot: insertion tasks sit at the origin, where the pedestal collider would be (cli._make_backend).
sim = IsaacBackend(cfg.cable, cfg.physics, render_resolutions=(cfg.render.full_resolution, (768, 768)),
                   robot=RobotSpec(model="none", base_collider=False))
from gymnasium.utils.env_checker import check_env  # noqa: E402

from cablerouting.bridge import SimInsertionWorld  # noqa: E402
from cablerouting.cli import _stage_source  # noqa: E402
from cablerouting.geometry.threading import evaluate_routing  # noqa: E402
from cablerouting.scenes.insertion import InsertionStartDistribution  # noqa: E402
from dloroute.rl.training import InsertionEnv  # noqa: E402

stages, stage_meta = _stage_source(sys.argv[1] if len(sys.argv) > 1 else "manifests/shapes_train.json")
print(f"[train-smoke] stage source {json.dumps(stage_meta)}", flush=True)
world = SimInsertionWorld(sim, cfg.family("primary10"), cfg.cable, cfg.motion,
                          InsertionStartDistribution(route_stage_prob=0.5,
                                                     stages=stages))
env = InsertionEnv(world, cfg.skills, cfg.obs, cfg.reward, seed=0)
check_env(env, skip_render_check=True)
print("[train-smoke] gymnasium check_env passed", flush=True)
rng = np.random.default_rng(0)
for ep in range(4):
    t = time.time()
    obs, info = env.reset(seed=100 + ep)
    start = evaluate_routing(world.clips, sim.cable_points(), []).summary()
    head = world.clips[-1].pose.to_local(sim.cable_points()[0])
    print(f"[train-smoke] ep {ep} start: routing {start['sequence']} partial {start['partial']}, "
          f"head in target frame ({head[0]:+.3f}, {head[1]:+.3f}) m", flush=True)
    rows = []
    for _ in range(2):
        obs, r, term, trunc, inf = env.step(rng.uniform(-1, 1, 7).astype(np.float32))
        rows.append({"r": round(r, 3), "term": term, "succ": inf["success"], "coll": inf["collided"],
                     "sd": round(inf["signed_dist"], 3), "terms": {k: round(v, 3) for k, v in inf["reward_terms"].items()}})
        if term or trunc:
            break
    print(f"[train-smoke] ep {ep} mode={info['mode']} scale={info['scale']:.2f} obs={obs.shape} "
          f"{time.time() - t:.1f}s {json.dumps(rows)}", flush=True)
sharp = tuple(st for st in stages if abs(st[1]) > 80)
if sharp:
    corner = SimInsertionWorld(sim, cfg.family("primary10"), cfg.cable, cfg.motion,
                               InsertionStartDistribution(route_stage_prob=1.0,
                                                          stages=sharp))
    crng = np.random.default_rng(1)
    for i in range(4):
        task = corner.reset_task(crng, 1.0)
        p, t = corner.clips[0].pose, corner.clips[1].pose
        seq = evaluate_routing(corner.clips, sim.cable_points(), []).summary()
        ok = seq["sequence"] == ["P+"] and not seq["partial"] and not seq["anomalies"]
        print(f"[train-smoke] sharp start {i}: turn {np.degrees(np.angle(np.exp(1j * (t.yaw - p.yaw)))):+.1f} deg, "
              f"start routing {seq['sequence']} partial {seq['partial']} -> {'OK' if ok else 'UNEXPECTED'}", flush=True)
from stable_baselines3 import SAC  # noqa: E402

model = SAC("MlpPolicy", env, learning_starts=4, batch_size=4, buffer_size=100, seed=0, device="cpu")
model.learn(total_timesteps=8)
print("[train-smoke] SAC plumbing OK; updates:", model._n_updates, flush=True)
sim.close()
