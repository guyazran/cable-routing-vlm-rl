"""The algorithm's only contract with the outside world.

A :class:`World` is whatever the algorithm acts in: the Isaac Sim environment
(``cablerouting.bridge.SimWorld``) or, later, a physical robot with perception.
The algorithm sees:

* a static :class:`SceneSpec`: clip IDs, poses and scales, and the cable radius
  (the paper's state observation includes clip pose and scale);
* the current :class:`CableState`: the ordered cable centreline and the clips it
  currently passes through. In simulation this is ground truth. The insertion
  policy and the reference controller read it, as in the paper's architecture and
  the protocol's reference. The VLM controller never reads it, except in the
  separately labelled true-progress diagnostic;
* rendered :class:`Views` (full scene and a crop around one clip) for the VLM.

It acts only through :meth:`World.pick_place`: grasp one centreline point, carry it
through planar waypoints, lower, release and settle. Collision checking, physics,
settling and rendering belong to the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from .geometry import Pose2D


@dataclass(frozen=True)
class ClipSpec:
    """A clip as the algorithm knows it. Local frame: +x is the insertion direction."""

    clip_id: str
    pose: Pose2D
    scale: float
    depth: float  # scaled tunnel length along the insertion axis

    @property
    def entry_x(self) -> float:
        return -self.depth / 2

    @property
    def exit_x(self) -> float:
        return self.depth / 2

    def floor_point(self) -> np.ndarray:
        """Centre of the entry face ("floor" in the paper's Fig. 3), table-plane xy."""
        return self.pose.to_world(np.array([self.entry_x, 0.0]))

    def ceil_point(self) -> np.ndarray:
        """Centre of the exit face ("ceiling"), table-plane xy."""
        return self.pose.to_world(np.array([self.exit_x, 0.0]))


@dataclass(frozen=True)
class SceneSpec:
    clips: tuple[ClipSpec, ...]
    cable_radius: float

    @property
    def clip_ids(self) -> list[str]:
        return [c.clip_id for c in self.clips]

    def clip(self, clip_id: str) -> ClipSpec:
        return next(c for c in self.clips if c.clip_id == clip_id)


@dataclass(frozen=True)
class Transit:
    """One pass of the cable through a clip's aperture."""

    clip_id: str
    direction: int  # +1 along the clip's insertion direction, -1 against it


@dataclass(frozen=True, eq=False)
class CableState:
    points: np.ndarray  # (n, 3) centreline, world frame, head tip first
    transits: tuple[Transit, ...]  # every clip transit, ordered from the tail to the head
    partial: Mapping[str, int]  # clips whose tunnel holds the head: +1 entered forward, -1 reversed


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float  # heading of the held cable tangent (towards the head), world frame


@dataclass(frozen=True)
class PickPlace:
    """One pick-and-place primitive: grasp, lift, carry through the waypoints, lower, release."""

    grasp_index: int  # index into CableState.points observed just before the command
    waypoints: tuple[Waypoint, ...]
    transport_height: float  # grasp-point height while carrying
    label: str = ""


@dataclass
class PrimitiveResult:
    collided: bool  # terminal robot/gripper collision under the world's collision rule
    collision_clips: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)  # world-specific execution log; archived, never interpreted


@dataclass
class Views:
    full: Any  # full-scene image
    crop: Any  # zoomed image centred on info["clip"]
    info: dict


class World(Protocol):
    scene: SceneSpec

    def state(self) -> CableState: ...

    def views(self, crop_clip: str) -> Views: ...

    def pick_place(self, cmd: PickPlace) -> PrimitiveResult: ...

    def grasp_collides(self, grasp_index: int) -> bool:
        """Whether descending open onto centreline point ``grasp_index`` and closing there would
        hit a clip. A feasibility query: nothing moves."""
        ...
