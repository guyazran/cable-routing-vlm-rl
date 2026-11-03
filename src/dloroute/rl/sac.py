"""SAC training of the insertion policy (stable-baselines3, MLP actor and critics).

The paper reports 6,200 training "steps" without defining them. Here one
environment step is one seven-parameter pick-and-place primitive (+ release and
settling); optimizer updates are counted separately in the training metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SACConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 200
    batch_size: int = 256
    gamma: float = 0.95
    gradient_steps: int = 1  # optimizer updates per environment step
    net_arch: tuple[int, ...] = (256, 256)


def latest_checkpoint(out: Path) -> tuple[Path, Path, int] | None:
    """The newest ``sac_<N>_steps.zip`` in ``out`` that has its replay buffer, with N."""
    best = None
    for z in out.glob("sac_*_steps.zip"):
        n = int(z.stem.split("_")[1])
        rb = out / f"sac_replay_buffer_{n}_steps.pkl"
        if rb.exists() and (best is None or n > best[2]):
            best = (z, rb, n)
    return best


def prefill(model, behavior, n_steps: int, noise: float, seed: int, demos: list | None = None) -> int:
    """Fill the replay buffer with ``n_steps`` real environment transitions from ``behavior`` plus
    Gaussian action noise (a guided warm-up instead of SB3's uniform-random one). Returns the steps.
    If ``demos`` is a list, (observation, noise-free behaviour action) pairs are appended to it."""
    import numpy as np

    rng = np.random.default_rng(seed)
    env = model.get_env()
    obs = env.reset()
    added = 0
    while added < n_steps:
        clean = np.stack([behavior(o) for o in obs])
        if demos is not None:
            demos += list(zip(np.asarray(obs, dtype=np.float32), np.clip(clean, -1, 1).astype(np.float32)))
        acts = np.clip(clean + rng.normal(0.0, noise, (env.num_envs, 7)), -1.0, 1.0)
        acts = acts.astype(np.float32)
        new_obs, rewards, dones, infos = env.step(acts)
        nxt = new_obs.copy()
        for i, d in enumerate(dones):
            if d:
                nxt[i] = infos[i]["terminal_observation"]
        model.replay_buffer.add(obs, nxt, acts, rewards, dones, infos)
        obs = new_obs
        added += env.num_envs
    model.num_timesteps += added
    model._last_obs = None  # learn() resets the environments before its first step
    return added


def behaviour_clone(model, demos: list, steps: int, batch: int, seed: int) -> float:
    """Fit the actor's deterministic action tanh(mean) to the demonstrated actions (MSE)."""
    import numpy as np
    import torch as th

    rng = np.random.default_rng(seed)
    obs = th.as_tensor(np.stack([d[0] for d in demos]))
    act = th.as_tensor(np.stack([d[1] for d in demos]))
    loss = th.zeros(())
    for _ in range(steps):
        idx = th.as_tensor(rng.integers(len(demos), size=batch))
        mean, _, _ = model.actor.get_action_dist_params(obs[idx])
        loss = ((th.tanh(mean) - act[idx]) ** 2).mean()
        model.actor.optimizer.zero_grad()
        loss.backward()
        model.actor.optimizer.step()
    return float(loss)


def train_sac(env, out: Path, total_steps: int, cfg: SACConfig = SACConfig(), seed: int = 0,
              checkpoint_every: int = 500, meta: dict | None = None, warmup_policy=None,
              warmup_steps: int = 0, warmup_noise: float = 0.15, bc_steps: int = 0,
              offline_steps: int = 0) -> dict:
    """Train on ``env`` (an InsertionEnv, or a VecEnv of them), saving checkpoints, the final model
    and metadata in ``out``.

    With n parallel environments every vectorised step adds n transitions, so the optimizer runs
    ``n * gradient_steps`` updates per vectorised step: updates per environment transition are the
    same as with one environment. Checkpoints are saved every ``checkpoint_every`` environment steps,
    each with its replay buffer. If ``out`` already holds a checkpoint, training resumes from the
    newest one (a preempted and requeued job); the environments' own step counters must already
    have been set to its step count (see :func:`resume_step`). With ``warmup_policy`` a fresh run
    first collects ``warmup_steps`` transitions from it (plus noise) instead of SB3's random
    warm-up; they count as environment steps.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecEnv, VecMonitor

    out.mkdir(parents=True, exist_ok=True)
    keys = ("success", "collided", "mode")
    n_envs = env.num_envs if isinstance(env, VecEnv) else 1
    resume = latest_checkpoint(out)
    segment = f"monitor_from{resume[2]}.csv" if resume else "monitor.csv"
    env = VecMonitor(env, str(out / segment), info_keywords=keys) if n_envs > 1 else \
        Monitor(env, str(out / segment), info_keywords=keys)
    if resume:
        model = SAC.load(resume[0], env=env, device="cpu", tensorboard_log=str(out / "tb"))
        model.load_replay_buffer(resume[1])
        meta = json.loads((out / "train_meta.json").read_text())
        meta.setdefault("resumed_at_steps", []).append(resume[2])
    else:
        model = SAC("MlpPolicy", env, learning_rate=cfg.learning_rate, buffer_size=cfg.buffer_size,
                    learning_starts=cfg.learning_starts, batch_size=cfg.batch_size, gamma=cfg.gamma, train_freq=1,
                    gradient_steps=cfg.gradient_steps * n_envs, policy_kwargs={"net_arch": list(cfg.net_arch)},
                    seed=seed, verbose=1, device="cpu", tensorboard_log=str(out / "tb"))
        meta = {**(meta or {}), "seed": seed, "env_steps": total_steps, "n_envs": n_envs,
                "step_semantics": "1 env step = 1 seven-parameter pick-and-place primitive (+ release and settling)",
                "sac": asdict(cfg), "started": time.time()}
    done_steps = resume[2] if resume else 0
    if not resume and warmup_policy is not None and warmup_steps > 0:
        demos: list = []
        done_steps = prefill(model, warmup_policy, warmup_steps, warmup_noise, seed, demos)
        meta["guided_warmup"] = {"steps": done_steps, "noise_std": warmup_noise,
                                 "behavior": getattr(warmup_policy, "__class__", type(warmup_policy)).__name__}
        if bc_steps:  # initialise the actor by behaviour cloning on the warm-up states
            meta["guided_warmup"]["bc"] = {"steps": bc_steps, "final_mse": behaviour_clone(
                model, demos, bc_steps, cfg.batch_size, seed)}
        if offline_steps:  # fit the critics (and actor) on the warm-up buffer before collecting more
            from stable_baselines3.common.logger import configure

            model.set_logger(configure(str(out / "tb_offline"), ["csv"]))
            model.train(gradient_steps=offline_steps, batch_size=cfg.batch_size)
            meta["guided_warmup"]["offline_gradient_steps"] = offline_steps
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    model.learn(total_timesteps=total_steps - done_steps, reset_num_timesteps=not (resume or done_steps),
                callback=CheckpointCallback(max(1, checkpoint_every // n_envs), str(out), "sac",
                                            save_replay_buffer=True))
    model.save(out / "sac_final")
    meta.update({"finished": time.time(), "optimizer_updates": int(model._n_updates)})
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def resume_step(out: Path) -> int:
    """Environment steps already done in ``out`` (0 for a fresh run)."""
    r = latest_checkpoint(out)
    return r[2] if r else 0
