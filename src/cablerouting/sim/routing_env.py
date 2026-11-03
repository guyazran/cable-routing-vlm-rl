"""Scene-level environment: saved-state resets, rendering, pick-and-place execution, grading.

Simulator independent: everything physical goes through the backend (see
sim.isaac_backend for the Isaac Sim implementation). The environment knows
nothing about the algorithm's skills, policies or controllers; it executes
pick-and-place primitives and grades routing with its own evaluator.
cablerouting.bridge exposes it to the algorithm as a dloroute World.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.clip import Clip
from ..geometry.threading import RoutingState, evaluate_routing
from ..robot.executor import ExecutionResult, Executor, MotionParams, PickPlace
from ..scenes.generator import Scene
from .camera_model import TopDownCamera, full_view_camera, zoom_camera
from .overlay import annotate


@dataclass(frozen=True)
class RenderConfig:
    full_resolution: tuple[int, int] = (1600, 1200)
    full_height: float = 2.0
    crop_size: float = 0.24
    crop_resolution: int = 768
    crop_height: float = 0.6


class RoutingEnv:
    def __init__(self, backend, scene: Scene, view_x, view_y, motion: MotionParams = MotionParams(),
                 render: RenderConfig = RenderConfig()):
        self.sim = backend
        self.scene = scene
        self.clips: list[Clip] = scene.clip_objects()
        self.by_id = {c.clip_id: c for c in self.clips}
        self.rc = render
        self.full_cam: TopDownCamera = full_view_camera(view_x, view_y, render.full_resolution, render.full_height)
        self.sim.load_scene(self.clips)
        self.executor = Executor(backend, self.clips, scene.cable_radius, motion)

    # --- state ----------------------------------------------------------------
    def reset(self, state: dict) -> None:
        self.sim.restore_state(state)

    def points(self) -> np.ndarray:
        return self.sim.cable_points()

    def routing(self, route: list[str]) -> RoutingState:
        return evaluate_routing(self.clips, self.points(), route)

    # --- observation ------------------------------------------------------------
    def render(self, crop_clip: str) -> tuple:
        full = annotate(self.sim.render(self.full_cam), self.clips, self.full_cam)
        zc = zoom_camera(self.by_id[crop_clip].pose.xy, self.rc.crop_size, self.rc.crop_resolution, self.rc.crop_height)
        crop = annotate(self.sim.render(zc), self.clips, zc)
        return full, crop, {"clip": crop_clip, "bounds": zc.world_bounds(), "resolution": zc.resolution}

    # --- actuation ----------------------------------------------------------------
    def pick_place(self, plan: PickPlace) -> ExecutionResult:
        return self.executor.pick_place(plan)

    def grasp_collides(self, point_index: int) -> bool:
        return self.executor.grasp_collides(point_index)
