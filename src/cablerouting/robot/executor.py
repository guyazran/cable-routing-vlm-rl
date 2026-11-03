"""Pick-and-place execution on any simulator backend (protocol Section 5).

One pick-and-place: free approach above the grasp point, vertical descent,
close and attach, simulated held transport through the via points, lower,
detach and open, free retreat to the home pose, then a fixed number of settling
steps. Free-space gripper motion does not touch the cable (fingers do not
collide with it) and is checked for clip collisions geometrically but not
physically simulated; every phase in which the gripper holds the cable is
simulated step by step. The collision rule is checked at every sampled gripper
pose of every phase, so a terminal collision is reported at its first
occurrence.

The executor is the robot side of the algorithm's pick-and-place command: it
knows nothing about skills, policies or controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..geometry.clip import Clip
from ..geometry.collision import colliding_clips
from ..geometry.frames import wrap_angle
from .gripper import GripperDims, gripper_boxes


class SimBackend(Protocol):
    """Minimal simulator surface used by the executor."""

    physics_dt: float

    def cable_points(self) -> np.ndarray: ...  # (n+1, 3) centreline, head tip first
    def point_segment(self, point_index: int) -> int: ...  # segment to attach for a centreline point
    def set_gripper_pose(self, pose: np.ndarray, opening: float) -> None: ...  # teleport, no physics
    def attach(self, segment: int) -> None: ...
    def detach(self) -> None: ...
    def step(self, n: int = 1, gripper_target: np.ndarray | None = None) -> None: ...


@dataclass(frozen=True)
class MotionParams:
    """Reconstruction choices for the shared pick-and-place primitive."""

    safe_height: float = 0.08  # grasp-point height for free moves (fingertips clear clip tops)
    # Franka 'ready' posture (grasp point 0.49 m up). Free moves from here stay above every
    # clip, so it does not change collision outcomes; observation renders hide the robot.
    home_pose: tuple[float, float, float, float] = (0.31, 0.0, 0.49, 0.0)
    hold_speed: float = 0.10  # m/s while holding the cable
    hold_yaw_rate: float = np.radians(90.0)  # rad/s while holding
    vertical_speed: float = 0.05  # m/s for held lift/lower
    free_sample_spacing: float = 0.005  # m between collision samples on free moves
    settle_steps: int = 240  # physics steps after release, identical for every skill
    grasp_close_margin: float = 0.002  # jaw opening = cable diameter + margin when closed


@dataclass
class Via:
    x: float
    y: float
    yaw: float  # heading of the held cable tangent (towards the head), world frame


@dataclass
class PickPlace:
    grasp_point: int  # centreline index, head first
    vias: list[Via]
    transport_height: float  # grasp-point height during held transport
    label: str = ""


@dataclass
class ExecutionResult:
    label: str
    collided: bool = False
    collision_clips: list[str] = field(default_factory=list)
    collision_phase: str | None = None
    physics_steps: int = 0
    grasp_xyz: list[float] | None = None
    trajectory: list[list[float]] = field(default_factory=list)  # held-phase gripper poses (subsampled)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def cable_tangent_yaw(points: np.ndarray, i: int) -> float:
    """Heading of the cable at centreline point i, pointing towards the head."""
    a = points[max(i - 1, 0)]
    b = points[min(i + 1, len(points) - 1)]
    d = a[:2] - b[:2]
    return float(np.arctan2(d[1], d[0]))


def grasp_collides(points: np.ndarray, point_index: int, clips: list[Clip], closed_width: float,
                   gripper: GripperDims = GripperDims()) -> bool:
    """Whether the gripper, open or closed at centreline point ``point_index`` (top-down, jaws across
    the cable tangent), overlaps a clip."""
    pose = np.array([*points[point_index][:3], cable_tangent_yaw(points, point_index)])
    return any(colliding_clips(gripper_boxes(pose, w, gripper), clips) for w in (gripper.max_opening, closed_width))


class Executor:
    def __init__(self, backend: SimBackend, clips: list[Clip], cable_radius: float,
                 params: MotionParams = MotionParams(), gripper: GripperDims = GripperDims()):
        self.sim = backend
        self.clips = clips
        self.r = cable_radius
        self.p = params
        self.g = gripper

    @property
    def closed_width(self) -> float:
        return 2 * self.r + self.p.grasp_close_margin

    def grasp_collides(self, point_index: int) -> bool:
        """Feasibility query on the current cable state; nothing moves."""
        return grasp_collides(self.sim.cable_points(), point_index, self.clips, self.closed_width, self.g)

    # --- collision helpers ---------------------------------------------------
    def _check(self, pose: np.ndarray, opening: float, res: ExecutionResult, phase: str) -> bool:
        hits = colliding_clips(gripper_boxes(pose, opening, self.g), self.clips)
        if hits and not res.collided:
            res.collided, res.collision_clips, res.collision_phase = True, hits, phase
        return bool(hits)

    def _free_move(self, a: np.ndarray, b: np.ndarray, opening: float, res: ExecutionResult, phase: str) -> bool:
        n = max(2, int(np.ceil(np.linalg.norm(b[:3] - a[:3]) / self.p.free_sample_spacing)) + 1)
        for t in np.linspace(0, 1, n):
            pose = a + t * np.append(b[:3] - a[:3], wrap_angle(b[3] - a[3]))
            if self._check(pose, opening, res, phase):
                self.sim.set_gripper_pose(pose, opening)
                return True
        self.sim.set_gripper_pose(b, opening)
        return False

    def _held_move(self, a: np.ndarray, b: np.ndarray, opening: float, res: ExecutionResult,
                   phase: str, vertical: bool = False) -> tuple[bool, np.ndarray]:
        dyaw = float(wrap_angle(b[3] - a[3]))
        dist = float(np.linalg.norm(b[:3] - a[:3]))
        speed = self.p.vertical_speed if vertical else self.p.hold_speed
        duration = max(dist / speed, abs(dyaw) / self.p.hold_yaw_rate, self.sim.physics_dt)
        n = int(np.ceil(duration / self.sim.physics_dt))
        for k in range(1, n + 1):
            t = k / n
            pose = a + t * np.append(b[:3] - a[:3], dyaw)
            if self._check(pose, opening, res, phase):
                return True, pose
            self.sim.step(1, gripper_target=pose)
            res.physics_steps += 1
            if k % 10 == 0 or k == n:
                res.trajectory.append([float(v) for v in pose])
        return False, b

    # --- primitive -----------------------------------------------------------
    def pick_place(self, plan: PickPlace) -> ExecutionResult:
        res = ExecutionResult(plan.label)
        pts = self.sim.cable_points()
        gp = pts[plan.grasp_point]
        yaw = cable_tangent_yaw(pts, plan.grasp_point)
        res.grasp_xyz = [float(v) for v in gp]
        open_w = self.g.max_opening
        closed_w = self.closed_width
        home = np.array(self.p.home_pose)
        above = np.array([gp[0], gp[1], self.p.safe_height, yaw])
        at = np.array([gp[0], gp[1], gp[2], yaw])

        # Free approach and descent (no physics: the gripper cannot affect the cable here).
        if self._free_move(home, above, open_w, res, "approach") or \
                self._free_move(above, at, open_w, res, "descend") or \
                self._check(at, closed_w, res, "close"):
            return self._abort(res)
        self.sim.set_gripper_pose(at, closed_w)
        self.sim.attach(self.sim.point_segment(plan.grasp_point))

        # Held transport: lift, via points, lower.
        cur = at
        targets = [(np.array([gp[0], gp[1], plan.transport_height, yaw]), True, "lift")]
        for i, v in enumerate(plan.vias):
            targets.append((np.array([v.x, v.y, plan.transport_height, v.yaw]), False, f"via{i + 1}"))
        last = plan.vias[-1] if plan.vias else None
        if last is not None:
            targets.append((np.array([last.x, last.y, self.r, last.yaw]), True, "lower"))
        for target, vertical, phase in targets:
            hit, cur = self._held_move(cur, target, closed_w, res, phase, vertical)
            if hit:
                return self._abort(res)

        # Release and free retreat.
        self.sim.detach()
        lifted = np.array([cur[0], cur[1], self.p.safe_height, cur[3]])
        if self._check(cur, open_w, res, "open") or \
                self._free_move(cur, lifted, open_w, res, "retreat") or \
                self._free_move(lifted, home, open_w, res, "home"):
            return self._abort(res, attached=False)
        self.sim.step(self.p.settle_steps)
        res.physics_steps += self.p.settle_steps
        return res

    def _abort(self, res: ExecutionResult, attached: bool = True) -> ExecutionResult:
        """Terminal collision: release the cable and leave the state as it is."""
        if attached:
            self.sim.detach()
        return res
