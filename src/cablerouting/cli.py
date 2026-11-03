"""Command-line entry point: ``python -m cablerouting <command> ...``.

Environment preparation:
  config-hash                          print the config hash (freeze identifier)
  gen-scenes   --family --pool --count  scene manifest with rejected-seed log
  make-dataset                          shape x length dataset: train/validation/test + screen/calibration subsets
  make-states  --manifest ... [--shard]  settle each scene's cable once, save and check states (Isaac)
  render-scene --manifest --scene-id    render full view and crops (Isaac; prompt development)

Experiment drivers (implemented, not yet run):
  train-insertion                       SAC training of the insertion policy (Isaac)
  eval-insertion                        single-clip evaluation of a checkpoint (Isaac)
  run-episode                           one routing episode (Isaac [+ vLLM])
  run-screen                            the 4-condition screen over a manifest (Isaac + vLLM)
  freeze       --policy --manifests     write configs/FROZEN (hashes of everything the screen uses)
  analyze                               Section 11-12 metrics and figures from run results

The algorithm (planner, controllers, skills, insertion policy) is the dloroute
package; this harness builds it from the config and connects it to the
simulator through cablerouting.bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

from .config import ROOT, load_config

STATE_DIR = ROOT / "manifests" / "states"


# ---------------------------------------------------------------- helpers

def _state_path(scene_id: str) -> Path:
    return STATE_DIR / f"{scene_id}.npz"


def _load_state(scene_id: str) -> dict:
    with np.load(_state_path(scene_id)) as z:
        return {k: z[k] for k in z.files}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _make_backend(cfg, robot: bool = True):
    """Routing scenes get the configured robot. Insertion tasks (robot=False) get none: they place the
    target clip at the origin, where the robot pedestal collider would sit on top of it. Insertion
    tasks never render, so they get a small viewport and small render products (GPU memory)."""
    from .sim.isaac_backend import IsaacBackend, RobotSpec

    import os

    if robot:
        res, viewport = (cfg.render.full_resolution, (cfg.render.crop_resolution,) * 2), (1280, 720)
    else:  # never rendered; CABLE_INSERT_VIEW=WxH overrides the size (benchmarks)
        w, h = map(int, os.environ.get("CABLE_INSERT_VIEW", "64x64").split("x"))
        res, viewport = ((w, h), (w, h)), (w, h)
    return IsaacBackend(cfg.cable, cfg.physics, render_resolutions=res,
                        robot=cfg.robot if robot else RobotSpec(model="none", base_collider=False),
                        park_pose=cfg.motion.home_pose, viewport=viewport)


def _policy(cfg, spec: str):
    from dloroute.rl.policies import HeuristicInsertPolicy, load_sac_policy

    from .geometry.clip import ClipDims

    if spec == "heuristic":
        return HeuristicInsertPolicy(cfg.obs, clip_depth=ClipDims().depth, region=cfg.skills.insertion_region)
    return load_sac_policy(spec)


def _versions(cfg, policy_spec: str) -> dict:
    from dloroute.planner.prompts import PromptBuilder

    v = {"config_hash": cfg.hash(), "policy": policy_spec,
         "versions": (ROOT / "configs" / "versions.yaml").read_text(),
         "timing_policy": "physics paused (not stepped) while waiting for any controller decision",
         "settle_steps_after_release": cfg.motion.settle_steps}
    if policy_spec != "heuristic" and Path(policy_spec).exists():
        v["policy_sha256"] = _file_hash(Path(policy_spec))
    try:
        v["prompt_hash"] = PromptBuilder(cfg.prompts).version_hash()
    except FileNotFoundError:
        v["prompt_hash"] = None
    return v


def _agent(cfg, kind: str, scene, route: list[str], episode_seed: int, policy_spec: str):
    """The algorithm for one episode: a high-level controller plus the shared skills."""
    from dloroute.agent import HierarchicalAgent
    from dloroute.skills import SkillLibrary

    skills = SkillLibrary(_policy(cfg, policy_spec), cfg.skills, cfg.obs, cfg.reward)
    if kind == "reference":
        from dloroute.controllers.reference import ReferenceController

        return HierarchicalAgent(ReferenceController(scene, route, cfg.skills, cfg.reference), skills)
    from dloroute.controllers.vlm import VLMController
    from dloroute.planner.client import PlannerClient
    from dloroute.planner.prompts import PromptBuilder

    ctrl = VLMController(PlannerClient(cfg.generation), PromptBuilder(cfg.prompts), route, scene.clip_ids,
                         episode_seed, privileged_progress=(kind == "vlm+progress"))
    return HierarchicalAgent(ctrl, skills)


# ---------------------------------------------------------------- commands

def cmd_config_hash(args, cfg) -> None:
    print(cfg.hash())


def cmd_gen_scenes(args, cfg) -> None:
    from .scenes.generator import config_hash, generate_pool, write_manifest

    fam = cfg.family(args.family)
    scenes, rejected = generate_pool(fam, args.pool, args.count)
    out = Path(args.out or ROOT / "manifests" / f"{fam.name}_{args.pool}.json")
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to regenerate (recorded layouts must not silently change)")
    write_manifest(out, fam, scenes, rejected)
    print(f"{len(scenes)} scenes, {len(rejected)} rejected seeds -> {out} (family hash {config_hash(fam)})")


def cmd_make_dataset(args, cfg) -> None:
    from .scenes.dataset import SPLITS, build_dataset, write_dataset

    d = cfg.dataset
    root = Path(args.out_dir or ROOT / "manifests")
    if any(root.glob(f"{d.name}_*.json")) and not args.force:
        sys.exit(f"{d.name} dataset manifests exist in {root}; pass --force to regenerate "
                 "(recorded layouts and splits must not silently change)")
    ds = build_dataset(cfg.families, d)
    paths = write_dataset(root, ds)
    width = max(len(s) for s in d.shapes)
    print(f"{'':{width}s}  " + " ".join(f"{'n=' + str(n):8s}" for n in d.n_clips) + " (train/validation/test per cell)")
    for shape in d.shapes:
        row = {c["n_clips"]: c for c in ds["cells"] if c["shape"] == shape}
        print(f"{shape:{width}s}  " + " ".join(f"{'/'.join(str(len(row[n][sp])) for sp in SPLITS):8s}"
                                              for n in d.n_clips))
    for k, p in paths.items():
        print(f"{k:14s} -> {p}")


def _physics_hash(cfg) -> str:
    """Hash of the settings a settled state depends on (cable, physics, robot colliders, clip geometry)."""
    import dataclasses

    from .geometry.clip import ClipDims

    raw = {k: cfg.raw.get(k) for k in ("cable", "physics", "robot")}
    raw["clip"] = dataclasses.asdict(ClipDims())
    blob = json.dumps(raw, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cmd_make_states(args, cfg) -> None:
    """Settle each scene's cable once and save it, then check every state (scenes/state_checks.py).

    Isaac starts only if a state is missing. With ``--shard k/n`` a process settles the missing
    scenes at manifest positions i with i % n == k and writes no index; a final run without
    ``--shard`` checks all states and writes ``<manifest>.index.json``. Exits with status 1 if
    any state fails a check. ``scripts/make_states_parallel.sh`` runs shards with a stall watchdog.
    """
    from .geometry.cable import link_points
    from .scenes.generator import load_manifest
    from .scenes.state_checks import LIMITS, check_settled

    shard = tuple(map(int, args.shard.split("/"))) if args.shard else None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    sim, n_failed = None, 0
    for manifest in map(Path, args.manifest):
        _, scenes = load_manifest(manifest)
        # Shard membership is by manifest position, so restarting a shard never overlaps another.
        todo = [s for i, s in enumerate(scenes) if (not shard or i % shard[1] == shard[0])
                and (args.force or not _state_path(s.scene_id).exists())]
        for s in todo:
            if sim is None:
                sim = _make_backend(cfg)
            sim.load_scene(s.clip_objects())
            sim.set_polyline(s.initial_polyline(cfg.cable.link_length))
            sim.step(args.settle_steps)
            st = sim.capture_state()
            np.savez_compressed(_state_path(s.scene_id), link_transforms=st["link_transforms"],
                                link_velocities=st["link_velocities"])
            print(f"{s.scene_id}: settled {args.settle_steps} steps", flush=True)
        if shard:
            print(f"{manifest.name}: shard {args.shard} settled {len(todo)} states (no index written)")
            continue
        index, checks = {}, {}
        for s in scenes:
            path = _state_path(s.scene_id)
            index[s.scene_id] = _file_hash(path)
            pts = link_points(_load_state(s.scene_id)["link_transforms"], cfg.cable.link_length)
            checks[s.scene_id] = check_settled(s, pts, cfg.cable.link_length, cfg.skills.insertion_region)
        failed = {k: v["failed"] for k, v in checks.items() if not v["passed"]}
        n_failed += len(failed)
        (STATE_DIR / f"{manifest.stem}.index.json").write_text(json.dumps(
            {"manifest_sha256": _file_hash(manifest), "settle_steps": args.settle_steps,
             "config_hash": cfg.hash(), "physics_hash": _physics_hash(cfg), "check_limits": LIMITS,
             "all_checks_passed": not failed, "states": index, "checks": checks}, indent=1))
        worst = max(checks.values(), key=lambda c: c["max_displacement_m"])
        print(f"{manifest.name}: {len(scenes)} states, {len(failed)} failed checks"
              f"{' ' + json.dumps(failed) if failed else ''}; max displacement {worst['max_displacement_m']:.4f} m")
    if sim is not None:
        sim.close()
    if n_failed:
        sys.exit(1)


def cmd_render_scene(args, cfg) -> None:
    from .scenes.generator import load_manifest
    from .sim.routing_env import RoutingEnv

    _, scenes = load_manifest(Path(args.manifest))
    scene = next(s for s in scenes if s.scene_id == args.scene_id)
    fam = cfg.family(scene.family)
    sim = _make_backend(cfg)
    env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
    env.reset(_load_state(scene.scene_id))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cid in args.crops or [scene.clips[0]["clip_id"]]:
        full, crop, _ = env.render(cid)
        crop.save(out / f"{scene.scene_id}_crop_{cid}.png")
    full.save(out / f"{scene.scene_id}_full.png")
    sim.close()


def _stage_source(spec: str) -> tuple[tuple, dict]:
    """``--stages``: a scene manifest whose stages the route-stage starts sample, or 'none' (family ranges)."""
    from .scenes.insertion import stage_pool

    if spec == "none":
        return (), {"stages": "none (family ranges)"}
    path = Path(spec) if Path(spec).is_absolute() else ROOT / spec
    stages = stage_pool(path)
    return stages, {"stages": str(path.relative_to(ROOT)), "stages_sha256": _file_hash(path), "n_stages": len(stages)}


def _insertion_world(cfg, sim, family: str, route_stage_prob: float, stages: tuple = ()):
    from .bridge import SimInsertionWorld
    from .scenes.insertion import InsertionStartDistribution

    dist = InsertionStartDistribution(route_stage_prob=route_stage_prob, stages=stages)
    return SimInsertionWorld(sim, cfg.family(family), cfg.cable, cfg.motion, dist)


class _InsertionEnvFactory:
    """Builds one insertion environment with its own Isaac process (a SubprocVecEnv worker).

    The curriculum switch counts this worker's own steps, so it is divided by the number of
    workers: workers step in lockstep, and the switch lands at the same total step count."""

    def __init__(self, config_path: str, family: str, route_stage_prob: float, stages_spec: str, seed: int,
                 n_envs: int, start_step: int = 0, rank: int = 0, absorbing_gamma: float | None = None):
        self.config_path, self.family, self.route_stage_prob = config_path, family, route_stage_prob
        self.stages_spec, self.seed, self.n_envs, self.start_step = stages_spec, seed, n_envs, start_step
        self.rank = rank
        self.absorbing_gamma = absorbing_gamma

    def __call__(self):
        import os

        from dloroute.rl.training import Curriculum, InsertionEnv

        # Stagger Kit start-ups: dozens starting at once contend for the shared extension cache.
        time.sleep(float(os.environ.get("CABLE_STAGGER_S", "2")) * self.rank)
        # Spread the workers' simulators over the job's GPUs (CUDA is initialised by Kit, after this).
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(gpus) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[self.rank % len(gpus)]
        cfg = load_config(self.config_path)
        stages, _ = _stage_source(self.stages_spec)
        sim = _make_backend(cfg, robot=False)
        cur = Curriculum()
        cur = Curriculum(-(-cur.switch_step // self.n_envs), cur.scale_initial, cur.scale_later)
        env = InsertionEnv(_insertion_world(cfg, sim, self.family, self.route_stage_prob, stages), cfg.skills,
                           cfg.obs, cfg.reward, cur, seed=self.seed, absorbing_gamma=self.absorbing_gamma)
        env.total_steps = self.start_step // self.n_envs  # resumed run: the curriculum continues where it was
        return env


def cmd_train_insertion(args, cfg) -> None:
    from dloroute.rl.sac import SACConfig, resume_step, train_sac

    _, stage_meta = _stage_source(args.stages)
    start = resume_step(Path(args.out))
    if start:
        print(f"[train] resuming {args.out} from its checkpoint at {start} env steps", flush=True)
    sac = SACConfig(learning_rate=args.lr, buffer_size=args.buffer, learning_starts=args.warmup,
                    batch_size=args.batch, gamma=args.gamma, gradient_steps=args.updates_per_step)
    meta = {"config_hash": cfg.hash(), "family": args.family, "route_stage_prob": args.route_stage_prob,
            "success": f"absorbing terminal state: reward r / (1 - {args.gamma})", **stage_meta}
    # A resumed segment draws fresh start states (seed offset by the resume step).
    factories = [_InsertionEnvFactory(args.config, args.family, args.route_stage_prob, args.stages,
                                      1000 * args.seed + rank + 7919 * start, args.n_envs, start, rank,
                                      absorbing_gamma=args.gamma)
                 for rank in range(args.n_envs)]
    if args.n_envs == 1:
        env = factories[0]()
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        env = SubprocVecEnv(factories, start_method="spawn")
    guide = _policy(cfg, "heuristic") if args.guided_warmup else None
    meta["warmup"] = (f"guided: {args.guided_warmup} steps of the heuristic policy + N(0, {args.guided_noise}) "
                      "action noise" if args.guided_warmup else f"SB3 uniform-random, {args.warmup} transitions")
    train_sac(env, Path(args.out), args.steps, sac, seed=args.seed, checkpoint_every=args.ckpt_every, meta=meta,
              warmup_policy=guide, warmup_steps=args.guided_warmup, warmup_noise=args.guided_noise,
              bc_steps=args.bc_steps, offline_steps=args.offline_steps)
    env.close()


def cmd_eval_insertion(args, cfg) -> None:
    from dloroute.rl.training import Curriculum, InsertionEnv

    stages, stage_meta = _stage_source(args.stages)
    sim = _make_backend(cfg, robot=False)
    env = InsertionEnv(_insertion_world(cfg, sim, "primary10", args.route_stage_prob, stages), cfg.skills, cfg.obs,
                       cfg.reward, Curriculum(switch_step=0, scale_later=(args.scale, args.scale)))
    policy = _policy(cfg, args.policy)
    rows = []
    for i in range(args.n):
        obs, info = env.reset(seed=args.seed_base + i)
        done, steps, last = False, 0, {}
        while not done:
            obs, _, term, trunc, last = env.step(policy(obs))
            steps += 1
            done = term or trunc
        rows.append({"scene": i, "seed": args.seed_base + i, "mode": info["mode"], "success": last["success"],
                     "collided": last["collided"], "signed_dist": last["signed_dist"], "steps": steps})
        print(json.dumps(rows[-1]), flush=True)
    sr = np.mean([r["success"] for r in rows])
    summary = {"n": len(rows), "success_rate": float(sr),
               "avg_signed_dist_cm": float(100 * np.mean([r["signed_dist"] for r in rows])),
               "avg_steps": float(np.mean([r["steps"] for r in rows])), "policy": args.policy,
               "route_stage_prob": args.route_stage_prob, **stage_meta}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary))
    sim.close()


def _run_one(cfg, sim, scene, controller_kind: str, n_targets: int, policy_spec: str, out: Path,
             episode_seed: int, per_clip_budget: int):
    from .bridge import scene_spec
    from .eval.episode import EpisodeConfig, run_episode
    from .sim.routing_env import RoutingEnv

    fam = cfg.family(scene.family)
    env = RoutingEnv(sim, scene, fam.view_x, fam.view_y, cfg.motion, cfg.render)
    agent = _agent(cfg, controller_kind, scene_spec(env.clips, scene.cable_radius), scene.route(n_targets),
                   episode_seed, policy_spec)
    return run_episode(env, agent, EpisodeConfig(n_targets, per_clip_budget), _load_state(scene.scene_id), out,
                       _versions(cfg, policy_spec) | {"episode_seed": episode_seed})


def cmd_run_episode(args, cfg) -> None:
    from .scenes.generator import load_manifest

    _, scenes = load_manifest(Path(args.manifest))
    scene = next(s for s in scenes if s.scene_id == args.scene_id)
    sim = _make_backend(cfg)
    res = _run_one(cfg, sim, scene, args.controller, args.targets, args.policy, Path(args.out),
                   args.seed, args.budget or cfg.study.per_clip_budget)
    print(json.dumps({k: res.__dict__[k] for k in ("scene_id", "controller", "n_targets", "success", "terminal",
                                                   "invocations", "physical_ops")}))
    sim.close()


def cmd_run_screen(args, cfg) -> None:
    """Four conditions per scene, condition order randomised per scene with the scheduling seed.

    Resumable: an episode with a result.json is skipped. A crashed (unscorable)
    episode is kept under ``attempt-<k>`` and rerun once, independent of outcome.
    """
    from .scenes.generator import load_manifest

    if not args.allow_unfrozen and not (ROOT / "configs" / "FROZEN").exists():
        sys.exit("configs/FROZEN not found: freeze the configuration (Section 9) before the screen, "
                 "or pass --allow-unfrozen for development runs")
    _, scenes = load_manifest(Path(args.manifest))
    scenes = scenes[: args.limit] if args.limit else scenes
    conditions = [(c, h) for c in args.controllers.split(",") for h in map(int, args.horizons.split(","))]
    rng = np.random.default_rng(cfg.study.scheduling_seed)
    schedule = []
    for i, s in enumerate(scenes):
        order = rng.permutation(len(conditions))
        # One episode seed per scene, shared by its conditions (paired RNG seeds, Section 10).
        schedule += [(s, *conditions[j], 1000 * i) for j in order]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "schedule.json").write_text(json.dumps(
        [{"scene": s.scene_id, "controller": c, "targets": h, "episode_seed": e} for s, c, h, e in schedule], indent=1))
    sim = _make_backend(cfg)
    for scene, ctrl, h, seed in schedule:
        ep_dir = root / scene.scene_id / f"{ctrl}_{h}"
        if (ep_dir / "result.json").exists():
            continue
        for attempt in range(2):
            try:
                _run_one(cfg, sim, scene, ctrl, h, args.policy, ep_dir, seed, cfg.study.per_clip_budget)
                break
            except Exception as e:  # infrastructure failure: archive and apply the declared rerun rule
                failed = ep_dir.parent / f"{ep_dir.name}.attempt-{attempt}"
                ep_dir.rename(failed)
                (failed / "failure.txt").write_text(f"{type(e).__name__}: {e}\n")
                print(f"[screen] {scene.scene_id} {ctrl}_{h}: crashed ({e}); archived {failed.name}", flush=True)
    sim.close()


def _eval_insertion_task(cfg, sim, spec: dict, log) -> dict:
    """One chunk of single-target insertion episodes; resumable per episode (seeded starts)."""
    from dloroute.rl.training import Curriculum, InsertionEnv

    out = Path(spec["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["seed"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    stages, stage_meta = _stage_source(spec.get("stages", "none"))
    scale = spec.get("scale", 1.0)
    env = InsertionEnv(_insertion_world(cfg, sim, "primary10", spec["route_stage_prob"], stages), cfg.skills,
                       cfg.obs, cfg.reward, Curriculum(switch_step=0, scale_later=(scale, scale)))
    policy = _policy(cfg, spec["policy"])
    with out.open("a") as fh:
        for seed in range(spec["seed_start"], spec["seed_start"] + spec["n"]):
            if seed in done:
                continue
            obs, info = env.reset(seed=seed)
            fin, steps, last = False, 0, {}
            while not fin:
                obs, _, term, trunc, last = env.step(policy(obs))
                steps += 1
                fin = term or trunc
            fh.write(json.dumps({"seed": seed, "mode": info["mode"], "scale": info["scale"],
                                 "success": bool(last["success"]), "collided": bool(last["collided"]),
                                 "signed_dist": float(last["signed_dist"]), "steps": steps}) + "\n")
            fh.flush()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    return {"status": "ok", "n": len(rows), "successes": sum(r["success"] for r in rows), **stage_meta}


def cmd_worker(args, cfg) -> None:
    """Pull units from shared queues until they are empty (eval/workqueue.py).

    ``--kinds episode`` runs routing episodes (robot backend); ``--kinds insertion_eval`` runs
    single-target insertion chunks (no robot). Planner episodes are taken only with ``--vlm-url``.
    After a crash the worker exits (status 3) so that a fresh process retries the unit: the launcher
    restarts it under the same worker id, and it takes back its own claim first.
    """
    import dataclasses
    import traceback

    from .eval.workqueue import MAX_ATTEMPTS, Queue, archive_attempt, worker_id
    from .scenes.generator import load_manifest

    queues = [Queue(Path(q)) for q in args.queue]
    wid = worker_id(args.index)
    kinds = set(args.kinds.split(","))
    if kinds - {"episode", "insertion_eval"} or len(kinds) != 1:
        sys.exit("--kinds: exactly one of episode, insertion_eval (one simulator backend per process)")
    kind = kinds.pop()
    if args.vlm_url:
        cfg = dataclasses.replace(cfg, generation=dataclasses.replace(cfg.generation, base_url=args.vlm_url))

    only = set(args.controllers.split(",")) if args.controllers else None

    def accept(spec: dict) -> bool:
        if spec["kind"] != kind:
            return False
        if kind == "episode" and only is not None and spec["controller"] not in only:
            return False
        return kind != "episode" or bool(args.vlm_url) or not spec["controller"].startswith("vlm")

    def log(msg: str) -> None:
        print(f"[worker {wid} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    sim, manifests, idle_since = None, {}, None
    while True:
        task = q = None
        for qq in queues:  # own unfinished claims first (requeued job), then queues in priority order
            own = qq.own_unfinished(wid)
            if own:
                task, q = own[0], qq
                break
        if task is None:
            for qq in queues:
                task = qq.claim_next(wid, accept)
                if task:
                    q = qq
                    break
        if task is None:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since > args.idle_exit:
                log("queues empty; exiting")
                break
            time.sleep(20)
            continue
        idle_since = None
        spec = q.spec(task)
        attempt = q.bump_attempts(task)
        if sim is None:
            sim = _make_backend(cfg, robot=(kind == "episode"))
        t0 = time.monotonic()
        if kind == "insertion_eval":
            log(f"{q.root.name}/{task}: insertion eval attempt {attempt}")
            summary = _eval_insertion_task(cfg, sim, spec, log)
            q.finish(task, {**summary, "worker": wid, "wall_s": time.monotonic() - t0})
            log(f"{task}: {summary['successes']}/{summary['n']} in {time.monotonic() - t0:.0f}s")
            continue
        out = Path(spec["out"])
        if out.exists() and (out / "result.json").exists() and attempt > 1:
            r = json.loads((out / "result.json").read_text())
            if r.get("terminal") not in ("error", "transport", ""):
                q.finish(task, {"status": "ok", "success": r["success"], "terminal": r["terminal"], "worker": wid})
                continue
        if out.exists():
            archive_attempt(out, f"unscorable: worker lost or crashed during attempt {attempt - 1} "
                                 "(preemption, node failure or exception); rerun from the saved initial state")
        if attempt > MAX_ATTEMPTS:
            q.finish(task, {"status": "failed", "reason": f"{MAX_ATTEMPTS} unscorable attempts", "worker": wid})
            log(f"{task}: giving up after {MAX_ATTEMPTS} attempts")
            continue
        if spec["manifest"] not in manifests:
            manifests[spec["manifest"]] = {s.scene_id: s for s in load_manifest(Path(spec["manifest"]))[1]}
        scene = manifests[spec["manifest"]][spec["scene_id"]]
        log(f"{q.root.name}/{task}: {spec['controller']} x{spec['targets']} attempt {attempt}")
        task_cfg = cfg
        if spec.get("overrides"):  # development variants; never physics, scenes or rendering
            if any(k.split(".")[0] in ("cable", "physics", "robot", "render", "scene_families", "dataset")
                   for k in spec["overrides"]):
                raise SystemExit(f"{task}: overrides may not touch the simulated environment")
            task_cfg = load_config(args.config, spec["overrides"])
            if args.vlm_url:
                task_cfg = dataclasses.replace(task_cfg, generation=dataclasses.replace(task_cfg.generation,
                                                                                       base_url=args.vlm_url))
        try:
            res = _run_one(task_cfg, sim, scene, spec["controller"], spec["targets"], spec["policy"], out,
                           spec["seed"], spec["budget"])
        except Exception as e:  # infrastructure failure: archive and let a fresh process rerun it
            archive_attempt(out, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            log(f"{task}: crashed ({type(e).__name__}: {e}); exiting for a fresh process")
            sys.exit(3)
        if res.terminal == "transport":
            archive_attempt(out, "unscorable: planner transport error (server unreachable); rerun")
            log(f"{task}: planner transport error; exiting for a fresh process")
            sys.exit(3)
        q.finish(task, {"status": "ok", "success": res.success, "terminal": res.terminal,
                        "invocations": res.invocations, "wall_s": res.wall_time_s, "worker": wid,
                        "host": socket.gethostname(), "vlm_url": args.vlm_url})
        log(f"{task}: success={res.success} terminal={res.terminal} inv={res.invocations} "
            f"{time.monotonic() - t0:.0f}s")
    if sim is not None:
        sim.close()


def cmd_queue(args, cfg) -> None:
    from .eval.workqueue import Queue, reap

    for qd in args.queue:
        q = Queue(Path(qd))
        if args.reap:
            print(f"{qd}: released {reap(q)}")
        print(f"{qd}: {json.dumps(q.status())}")


def code_hash() -> str:
    """Hash of the system's source: the algorithm (src/dloroute) and the environment and harness
    (src/cablerouting), excluding this command-line tooling file."""
    h = hashlib.sha256()
    for f in sorted(list((ROOT / "src" / "dloroute").rglob("*.py")) + list((ROOT / "src" / "cablerouting").rglob("*.py"))):
        if f.name == "cli.py" and f.parent.name == "cablerouting":
            continue
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def cmd_freeze(args, cfg) -> None:
    """Write configs/FROZEN: the immutable identifiers the screen will run under (Section 9)."""
    from dloroute.planner.prompts import PromptBuilder

    frozen = ROOT / "configs" / "FROZEN"
    if frozen.exists() and not args.force:
        sys.exit(f"{frozen} already exists")
    manifests = {m: _file_hash(Path(m)) for m in args.manifests}
    for m in args.manifests:
        idx = STATE_DIR / f"{Path(m).stem}.index.json"
        if not idx.exists():
            sys.exit(f"no settled states for {m}; run make-states first")
        rec = json.loads(idx.read_text())
        if rec.get("physics_hash") != _physics_hash(cfg) or rec.get("manifest_sha256") != _file_hash(Path(m)):
            sys.exit(f"settled states for {m} predate the current physics settings or manifest; rerun make-states --force")
        if not rec.get("all_checks_passed"):
            sys.exit(f"settled states for {m} failed their checks; see {idx}")
        manifests[str(idx)] = _file_hash(idx)
    record = {"config_hash": cfg.hash(), "prompt_hash": PromptBuilder(cfg.prompts).version_hash(),
              "code_sha256": code_hash(), "policy": args.policy, "policy_sha256": _file_hash(Path(args.policy)),
              "manifests": manifests, "versions": (ROOT / "configs" / "versions.yaml").read_text(),
              "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.calibration:  # the frozen system must be the one that passed calibration
        cal = json.loads(Path(args.calibration).read_text())
        diff = {k: (cal.get(k), record[k]) for k in ("code_sha256", "config_hash", "prompt_hash", "policy_sha256")
                if cal.get(k) != record[k]}
        if diff:
            sys.exit(f"system differs from the calibrated one: {diff}")
        record["calibration_system"] = args.calibration
    frozen.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "versions"}, indent=1))


def cmd_analyze(args, cfg) -> None:
    from .eval.analysis import analyze_screen

    report = analyze_screen(Path(args.runs), Path(args.out), cfg.study.per_clip_budget, seed=args.seed)
    print(json.dumps(report["primary"], indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cablerouting", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config-hash")
    p = sub.add_parser("gen-scenes")
    p.add_argument("--family", default="primary10")
    p.add_argument("--pool", choices=["development", "calibration", "evaluation"], required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("make-dataset")
    p.add_argument("--out-dir", help="default: manifests/")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("make-states")
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--settle-steps", type=int, default=480)
    p.add_argument("--shard", help="k/n: settle missing states at manifest positions i % n == k; no index written")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("render-scene")
    p.add_argument("--manifest", required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--crops", nargs="*")
    p.add_argument("--out", default=str(ROOT / "runs" / "renders"))
    p = sub.add_parser("train-insertion")
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="primary10")
    p.add_argument("--steps", type=int, default=6200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--updates-per-step", dest="updates_per_step", type=int, default=1)
    p.add_argument("--route-stage-prob", type=float, default=0.5)
    p.add_argument("--stages", default="manifests/shapes_train.json",
                   help="manifest whose stages the route-stage starts sample (official train split), or 'none'")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--guided-warmup", dest="guided_warmup", type=int, default=0,
                   help="fresh runs: first collect this many transitions with the heuristic policy + noise")
    p.add_argument("--guided-noise", dest="guided_noise", type=float, default=0.15)
    p.add_argument("--bc-steps", dest="bc_steps", type=int, default=0,
                   help="with --guided-warmup: behaviour-clone the actor on the warm-up states first")
    p.add_argument("--offline-steps", dest="offline_steps", type=int, default=0,
                   help="with --guided-warmup: SAC gradient steps on the warm-up buffer before collecting more")
    p.add_argument("--n-envs", dest="n_envs", type=int, default=1,
                   help="parallel environments, one Isaac process each (same updates per transition)")
    p = sub.add_parser("eval-insertion")
    p.add_argument("--policy", required=True, help="checkpoint path or 'heuristic'")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=9_000_000)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--route-stage-prob", type=float, default=0.0)
    p.add_argument("--stages", default="none",
                   help="manifest whose stages route-stage starts sample (e.g. the validation split), or 'none'")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run-episode")
    p.add_argument("--manifest", required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--controller", choices=["vlm", "reference", "vlm+progress"], required=True)
    p.add_argument("--targets", type=int, choices=[5, 10], required=True)
    p.add_argument("--policy", required=True, help="checkpoint path or 'heuristic'")
    p.add_argument("--budget", type=int, help="per-clip budget b (default: config)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run-screen")
    p.add_argument("--manifest", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--controllers", default="vlm,reference")
    p.add_argument("--horizons", default="5,10")
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--queue", nargs="+", required=True, help="queue directories, highest priority first")
    p.add_argument("--index", type=int, default=0, help="worker index within its Slurm job")
    p.add_argument("--kinds", default="episode", help="episode | insertion_eval")
    p.add_argument("--vlm-url", dest="vlm_url", help="planner server; without it planner episodes are skipped")
    p.add_argument("--idle-exit", dest="idle_exit", type=float, default=600.0)
    p.add_argument("--controllers", help="episode workers: take only these controllers (comma list)")
    p = sub.add_parser("queue")
    p.add_argument("--queue", nargs="+", required=True)
    p.add_argument("--reap", action="store_true", help="release claims whose Slurm job is gone")
    p = sub.add_parser("freeze")
    p.add_argument("--policy", required=True, help="frozen insertion checkpoint")
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--calibration", help="system.json recorded at calibration start; freeze refuses a different system")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=12345)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"config-hash": cmd_config_hash, "gen-scenes": cmd_gen_scenes, "make-dataset": cmd_make_dataset,
     "make-states": cmd_make_states,
     "render-scene": cmd_render_scene, "train-insertion": cmd_train_insertion,
     "eval-insertion": cmd_eval_insertion, "run-episode": cmd_run_episode, "run-screen": cmd_run_screen,
     "freeze": cmd_freeze, "analyze": cmd_analyze, "worker": cmd_worker, "queue": cmd_queue}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
