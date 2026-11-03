"""Insertion MDP for training and evaluating the policy (Gymnasium).

One env step = one 7-parameter pick-and-place primitive in the target clip's
frame, followed by release and settling; an episode ends on a collision, after
``local_horizon`` (paper: 7) steps, or on insertion success if
``terminate_on_success`` (evaluation; the Insert skill also stops there).

Training treats success as an absorbing state (``absorbing_gamma``; deviation
R11): the episode ends there, as the Insert skill does, and the final reward is
the state's Eq. 1-4 reward continued forever, r / (1 - gamma). Eq. 1-4 reward
states (rope_in / rope_out indicators, distance and flatness terms), all
non-negative apart from the collision and horizon penalties; a plain terminal
success would cut that stream, and with time-limit truncation bootstrapped as
non-terminal an agent that never inserts would value its states above an
insertion.

The start-state distribution belongs to the world (:class:`InsertionTaskWorld`);
this module owns the MDP: observation, action decoding, Eq. 1-4 reward,
termination, and the paper's clip-scale curriculum: U[1.0, 2.2] for the first
``switch_step`` env steps, then U[0.9, 1.5].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import gymnasium as gym
import numpy as np

from ..interface import CableState, PickPlace, PrimitiveResult, SceneSpec
from ..skills import SkillParams, decode_insert_action, insert_command
from ..state import head_metrics
from .observation import ObsSpec, observation
from .reward import RewardParams, insertion_success, reward


@dataclass
class InsertionTask:
    target: str
    info: dict = field(default_factory=dict)  # world-specific, e.g. the start-state mode


class InsertionTaskWorld(Protocol):
    scene: SceneSpec  # refreshed by reset_task

    def reset_task(self, rng: np.random.Generator, clip_scale: float) -> InsertionTask: ...

    def state(self) -> CableState: ...

    def pick_place(self, cmd: PickPlace) -> PrimitiveResult: ...


@dataclass(frozen=True)
class Curriculum:
    switch_step: int = 1600
    scale_initial: tuple[float, float] = (1.0, 2.2)
    scale_later: tuple[float, float] = (0.9, 1.5)

    def scale_range(self, env_steps: int) -> tuple[float, float]:
        return self.scale_initial if env_steps < self.switch_step else self.scale_later


class InsertionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, world: InsertionTaskWorld, skills: SkillParams = SkillParams(), obs_spec: ObsSpec = ObsSpec(),
                 rp: RewardParams = RewardParams(), curriculum: Curriculum = Curriculum(), seed: int = 0,
                 terminate_on_success: bool = True, absorbing_gamma: float | None = None):
        self.world = world
        self.sp, self.obs_spec, self.rp, self.curriculum = skills, obs_spec, rp, curriculum
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_spec.dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self.terminate_on_success = terminate_on_success
        self.absorbing_gamma = absorbing_gamma
        self.total_steps = 0
        self.t = 0
        self.target = None
        self.task_info: dict = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        lo, hi = self.curriculum.scale_range(self.total_steps)
        scale = float(self.np_random.uniform(lo, hi))
        task = self.world.reset_task(self.np_random, scale)
        self.target = self.world.scene.clip(task.target)
        self.task_info = task.info
        self.t = 0
        st = self.world.state()
        return observation(st.points, self.target, head_metrics(self.target, st).rope_in, self.obs_spec), \
            {**task.info, "scale": scale}

    def step(self, action):
        self.t += 1
        self.total_steps += 1
        act = decode_insert_action(action, self.target, self.obs_spec.n_points, self.sp)
        height = self.world.scene.cable_radius + self.sp.insert_clearance
        res = self.world.pick_place(insert_command(act, self.target, self.obs_spec, height))
        st = self.world.state()
        m = head_metrics(self.target, st)
        terms = reward(st.points, self.target, m, res.collided, self.t, self.obs_spec, self.rp,
                       self.sp.insertion_region)
        success = insertion_success(m, self.rp)
        terminated = res.collided or (success and self.terminate_on_success)
        truncated = not terminated and self.t >= self.sp.local_horizon
        info = {**self.task_info, "success": success, "collided": res.collided, "signed_dist": m.signed_dist,
                "reward_terms": terms.__dict__, "execution": res.details}
        obs = observation(st.points, self.target, m.rope_in, self.obs_spec)
        r = float(terms.total)
        if success and not res.collided and self.absorbing_gamma is not None:  # a success with a collision is a collision
            r /= 1.0 - self.absorbing_gamma  # absorbing success state: its reward, discounted forever
            info["absorbing_reward"] = r
        return obs, r, terminated, truncated, info

    def close(self):
        close = getattr(self.world, "close", None)
        if close is not None:
            close()
