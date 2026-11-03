"""Connects the algorithm (``dloroute``) to the simulated environment.

This module and the experiment harness (``eval``, ``cli``, ``config``) are the
only code that imports both packages: ``dloroute`` never imports
``cablerouting``, and the environment core (``geometry``, ``sim``, ``scenes``,
``robot``) never imports ``dloroute`` (tests/test_separation.py).

* :class:`SimWorld` is a :class:`dloroute.interface.World` over a routing episode.
* :class:`SimInsertionWorld` is a :class:`dloroute.rl.training.InsertionTaskWorld`
  over single-target insertion tasks.

The cable state handed to the algorithm is simulator ground truth: the
centreline and the transits found by the environment's geometric evaluator.
"""

from __future__ import annotations

import numpy as np

from dloroute import geometry as G
from dloroute.interface import (CableState, ClipSpec, PickPlace, PrimitiveResult, SceneSpec, Transit, Views)
from dloroute.rl.training import InsertionTask

from .geometry.clip import Clip
from .geometry.threading import evaluate_routing
from .robot import executor as X
from .scenes.generator import SceneFamilyConfig
from .scenes.insertion import InsertionStartDistribution, sample_insertion_scene
from .sim.isaac_backend import CableSpec
from .sim.routing_env import RoutingEnv


def clip_spec(c: Clip) -> ClipSpec:
    return ClipSpec(c.clip_id, G.Pose2D(c.pose.x, c.pose.y, c.pose.yaw), c.scale, c.dims.depth)


def scene_spec(clips: list[Clip], cable_radius: float) -> SceneSpec:
    return SceneSpec(tuple(clip_spec(c) for c in clips), cable_radius)


def cable_state(clips: list[Clip], points: np.ndarray) -> CableState:
    r = evaluate_routing(clips, points, [])
    return CableState(np.asarray(points, dtype=float), tuple(Transit(t.clip_id, t.direction) for t in r.sequence),
                      {k: v.partial for k, v in r.clips.items() if v.partial})


def to_plan(cmd: PickPlace) -> X.PickPlace:
    return X.PickPlace(cmd.grasp_index, [X.Via(w.x, w.y, w.yaw) for w in cmd.waypoints], cmd.transport_height,
                       cmd.label)


def to_result(r: X.ExecutionResult) -> PrimitiveResult:
    return PrimitiveResult(r.collided, list(r.collision_clips), r.to_dict())


class SimWorld:
    def __init__(self, env: RoutingEnv):
        self.env = env
        self.scene = scene_spec(env.clips, env.scene.cable_radius)

    def state(self) -> CableState:
        return cable_state(self.env.clips, self.env.points())

    def views(self, crop_clip: str) -> Views:
        full, crop, info = self.env.render(crop_clip)
        return Views(full, crop, info)

    def pick_place(self, cmd: PickPlace) -> PrimitiveResult:
        return to_result(self.env.pick_place(to_plan(cmd)))

    def grasp_collides(self, grasp_index: int) -> bool:
        return self.env.grasp_collides(grasp_index)


class SimInsertionWorld:
    def __init__(self, backend, family: SceneFamilyConfig, cable: CableSpec, motion: X.MotionParams = X.MotionParams(),
                 dist: InsertionStartDistribution = InsertionStartDistribution()):
        if getattr(backend, "robot", None) is not None:
            raise ValueError("insertion tasks put the target clip at the origin, inside the robot pedestal "
                             "collider; use a backend without the robot (cli._make_backend(cfg, robot=False))")
        self.sim = backend
        self.family, self.cable, self.motion, self.dist = family, cable, motion, dist
        self.clips: list[Clip] = []
        self.scene = SceneSpec((), cable.radius)
        self.executor: X.Executor | None = None

    def reset_task(self, rng: np.random.Generator, clip_scale: float) -> InsertionTask:
        s = sample_insertion_scene(rng, clip_scale, self.family, self.cable.length, self.cable.link_length,
                                   self.cable.radius, self.dist)
        self.sim.load_scene(s.clips)
        self.sim.set_polyline(s.polyline)
        self.sim.step(self.dist.settle_steps)
        self.clips = s.clips
        self.scene = scene_spec(s.clips, self.cable.radius)
        self.executor = X.Executor(self.sim, s.clips, self.cable.radius, self.motion)
        return InsertionTask(s.target.clip_id, {"mode": s.mode})

    def state(self) -> CableState:
        return cable_state(self.clips, self.sim.cable_points())

    def pick_place(self, cmd: PickPlace) -> PrimitiveResult:
        return to_result(self.executor.pick_place(to_plan(cmd)))

    def close(self) -> None:
        self.sim.close()
